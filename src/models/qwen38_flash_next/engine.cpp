#include "src/models/qwen38_flash_next/engine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <type_traits>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/wait_policy.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

// gfx1151 pp4096 at depths 0/4096: the 512/1024/2048/4096 sweep favored
// 2048; larger chunks used more scratch without improving throughput.
constexpr std::uint32_t kPrefillChunkTokens = 2048;

void AssignError(std::string* error_msg, std::string_view message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

constexpr std::array<char, 8> kSessionSnapshotMagic{'Q', 'F', 'N', 'S',
                                                    'E', 'S', 'S', '1'};

/// Host-side session fields ahead of the executor payload: the token
/// history and the logits of the last token.
struct SessionSnapshotHeader {
  std::array<char, 8> magic;
  std::uint32_t version;
  std::uint32_t vocab_size;
  std::uint32_t token_count;
  std::uint32_t hidden_rows;
  std::uint64_t executor_bytes;
  MtpLengthState draft_policy;
  std::uint32_t image_identity_bytes;
  std::uint32_t policy_concurrency;
  std::uint32_t reserved{0};
};
static_assert(std::is_trivially_copyable_v<SessionSnapshotHeader>);

std::uint64_t SessionSnapshotHostBytes(std::uint32_t token_count,
                                       std::uint32_t vocab_size,
                                       std::size_t image_bytes) {
  return sizeof(SessionSnapshotHeader) + image_bytes +
         std::uint64_t{token_count} * sizeof(std::int32_t) +
         std::uint64_t{vocab_size} * sizeof(float);
}

}  // namespace

struct SessionSnapshotData {
  // Last release follows destruction of every descriptor vector and payload.
  std::shared_ptr<const void> metadata;
  std::weak_ptr<Model> model;
  std::shared_ptr<const snapshot::Buffer> host;
  std::shared_ptr<const snapshot::Storage> executor;
  std::vector<SnapshotStorageOwner> owners;

  bool Extends(std::span<const std::int32_t> tokens,
               std::span<const std::uint8_t> identity) const {
    SessionSnapshotHeader h{};
    std::memcpy(&h, host->bytes().data(), sizeof(h));
    const auto image = host->bytes().subspan(sizeof(h), h.image_identity_bytes);
    const auto previous = host->bytes().subspan(
        sizeof(h) + h.image_identity_bytes,
        std::size_t{h.token_count} * sizeof(std::int32_t));
    return snapshot::ExactPrefix<std::uint8_t>(
        previous,
        {reinterpret_cast<const std::uint8_t*>(tokens.data()),
         tokens.size_bytes()},
        image, identity);
  }
};

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> m(new Model());
  if (options.decode_concurrency == 0 || options.decode_concurrency > 8) {
    AssignError(error_msg, "decode concurrency must be between one and eight");
    return nullptr;
  }
  if (options.max_draft_tokens == 0) {
    AssignError(error_msg, "draft token limit must be positive");
    return nullptr;
  }
  if (!options.mtp_model_path.empty() &&
      options.max_draft_tokens > kMaxMtpDraftTokens) {
    AssignError(error_msg,
                "Flash-Next MTP supports at most seven draft tokens");
    return nullptr;
  }
  m->options_ = options;
  m->reader_ = core::GgufReader::OpenFile(model_path, error_msg);
  if (!m->reader_) {
    return nullptr;
  }
  auto weights = ModelWeights::Bind(*m->reader_, error_msg);
  if (!weights) {
    return nullptr;
  }
  m->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  const Config& c = m->weights_->config;
  if (options.max_context == 0 || options.max_context > c.context_length) {
    AssignError(error_msg, "context exceeds the model's " +
                               std::to_string(c.context_length) + " tokens");
    return nullptr;
  }
  try {
    m->vision_ = qwen::vision::Encoder::Open(
        model_path, options.vision_model_path, c.hidden_size);
  } catch (const std::exception& e) {
    AssignError(error_msg, e.what());
    return nullptr;
  }
  m->tokenizer_ =
      tokenization::QwenTokenizer::CreateFromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  if (c.ple_layer >= 0) {
    const auto& t = m->weights_->ple_table;
    m->ngram_ = NgramTable::Open(
        m->reader_->GetMappedRegions()[t.shard].file_descriptor, t.file_offset,
        t.rows, c.ple_head_dim, t.type, error_msg);
    if (!m->ngram_) {
      return nullptr;
    }
  }
  if (!options.mtp_model_path.empty()) {
    m->mtp_reader_ =
        core::GgufReader::OpenFile(options.mtp_model_path, error_msg);
    if (!m->mtp_reader_) {
      return nullptr;
    }
    auto mtp = MtpWeights::Bind(*m->mtp_reader_, c, error_msg);
    if (!mtp) {
      return nullptr;
    }
    m->mtp_weights_ = std::make_unique<MtpWeights>(std::move(*mtp));
  }
  m->device_ = rocm::DeviceModel::Upload(*m->weights_, *m->reader_,
                                         m->mtp_weights_.get(),
                                         m->mtp_reader_.get(), error_msg);
  if (!m->device_) {
    return nullptr;
  }
  rocm::Executor::Options exec;
  exec.max_batch = m->PrefillCapacity();
  exec.max_logit_rows =
      m->mtp_weights_
          ? static_cast<std::uint32_t>(std::min<std::uint64_t>(
                exec.max_batch, std::uint64_t{options.max_draft_tokens} + 1))
          : 1;
  exec.max_speculative = exec.max_logit_rows;
  exec.history_budget_bytes = options.history_budget_bytes;
  exec.kv_quant = options.kv_quant;
  exec.history_logger = options.history_event_log;
  m->executor_ =
      rocm::Executor::Create(*m->device_, m->ngram_.get(), exec, error_msg);
  if (!m->executor_) {
    return nullptr;
  }
  return m;
}

std::unique_ptr<Session> Model::CreateSession(core::SessionMode mode,
                                              std::uint32_t max_context,
                                              std::string* error_msg) {
  if (max_context == 0 || max_context > options_.max_context) {
    AssignError(error_msg, "session context is outside the model limits");
    return nullptr;
  }
  auto native = executor_->CreateSession(mode, max_context, error_msg);
  if (!native) {
    return nullptr;
  }
  return std::unique_ptr<Session>(
      new Session(shared_from_this(), std::move(native)));
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text) const {
  std::vector<std::int32_t> out;
  for (auto id : tokenizer_->Encode(text)) {
    out.push_back(static_cast<std::int32_t>(id));
  }
  return out;
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<tokenization::TokenId> ids(tokens.begin(), tokens.end());
  return tokenizer_->Decode(ids);
}

std::string Model::TokenText(std::int32_t token) const {
  return tokenizer_->DecodeTokenCopy(static_cast<tokenization::TokenId>(token));
}

std::int32_t Model::EosToken() const noexcept {
  return static_cast<std::int32_t>(tokenizer_->GetEosTokenId());
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return token == EosToken() ||
         token == static_cast<std::int32_t>(tokenizer_->GetPadTokenId());
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab_size;
}

std::uint32_t Model::PrefillCapacity() const noexcept {
  return std::min(kPrefillChunkTokens, options_.max_context);
}

bool Model::HasMtp() const noexcept {
  return device_->has_mtp();
}

std::string Model::ModelName() const {
  return std::string(reader_->GetMetadataString("general.name")
                         .value_or("Qwen3.8-Flash-Next"));
}

const Config& Model::config() const noexcept {
  return weights_->config;
}

std::size_t Model::ResidentBytes() const noexcept {
  return device_->resident_bytes() + (vision_ ? vision_->ResidentBytes() : 0);
}

std::size_t Model::SessionBytes(core::SessionMode mode,
                                std::uint32_t context) const noexcept {
  const std::size_t vision =
      vision_ ? std::size_t{context} * (config().hidden_size * sizeof(float) +
                                        3 * sizeof(std::int32_t)) +
                    64
              : 0;
  return executor_->SessionBytes(mode, context,
                                 mode == core::SessionMode::kSpeculative
                                     ? executor_->max_speculative() - 1
                                     : 0) +
         vision;
}

std::size_t Model::ElasticHistoryBytes(
    core::SessionMode mode, std::uint32_t context) const noexcept {
  return executor_->ElasticHistoryBytes(mode, context);
}

std::uint32_t Model::history_initial_positions() const noexcept {
  return executor_->history_initial_positions();
}

std::size_t Model::DeferredScratchBytes() const {
  return executor_->DeferredScratchBytes();
}

std::size_t Session::AllocatedBytes() const noexcept {
  return session_->AllocatedBytes();
}

std::size_t Session::HistoryBytes() const noexcept {
  return session_->HistoryBytes();
}

bool Session::MtpEnabled() const noexcept {
  return session_->mtp_enabled();
}

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::Session> session)
    : model_(std::move(model)),
      session_(std::move(session)),
      draft_length_(model_->options_.max_draft_tokens,
                    model_->DecodeConcurrency()) {
  logits_.resize(model_->VocabSize());
}

Session::~Session() = default;

std::uint32_t Session::Position() const noexcept {
  return session_->position();
}
std::uint32_t Session::ContextSize() const noexcept {
  return session_->max_context();
}

void Session::Reset() {
  snapshot_parent_.reset();
  valid_ = false;
  session_->Reset();
  tokens_.clear();
  hidden_base_ = 0;
  draft_length_.Reset();
  single_policy_.Reset();
  lookup_cache_.Reset();
  model_->executor_->MtpRewind(*session_, 0);
  valid_ = true;
}

void Session::SetCancellationCheck(std::function<bool()> check) {
  session_->SetCancellationCheck(std::move(check));
}

void Session::ConfigureVision(
    std::shared_ptr<const qwen::vision::Prompt> prompt) {
  const auto identity =
      prompt ? prompt->cache_identity : std::vector<std::uint8_t>{};
  if (!tokens_.empty() && identity != image_identity_)
    Reset();
  const bool was_valid = valid_;
  valid_ = false;
  session_->ConfigureVision(std::move(prompt), model_->vision_,
                            model_->executor_->stream());
  image_identity_ = identity;
  valid_ = was_valid;
}

std::uint32_t Session::KeptHiddenRows() const noexcept {
  return MtpEnabled()
             ? static_cast<std::uint32_t>(tokens_.size() - hidden_base_)
             : 0;
}

std::uint64_t Session::SnapshotBytes() const {
  if (!valid_)
    return 0;
  return SessionSnapshotHostBytes(static_cast<std::uint32_t>(tokens_.size()),
                                  model_->VocabSize(), image_identity_.size()) +
         model_->executor_->SnapshotBytes(*session_, KeptHiddenRows());
}

std::uint64_t Session::SnapshotAllocationBytes() const {
  if (!valid_ || tokens_.empty())
    return 0;
  const auto host_bytes =
      SessionSnapshotHostBytes(static_cast<std::uint32_t>(tokens_.size()),
                               model_->VocabSize(), image_identity_.size());
  const auto regions = 5 * model_->config().num_layers + 12;
  const auto owners = regions * snapshot::Storage::kMaxSegments + 3;
  return host_bytes + sizeof(snapshot::Buffer) + sizeof(SessionSnapshotData) +
         sizeof(SessionSnapshot) + 1 + owners * sizeof(SnapshotStorageOwner) +
         model_->executor_->SnapshotAllocationBytes(*session_,
                                                    KeptHiddenRows());
}

std::uint64_t Session::SnapshotIncrementalBytes() const {
  const auto full = SnapshotAllocationBytes();
  if (!valid_ || tokens_.empty())
    return 0;
  auto parent = snapshot_parent_.lock();
  if (!parent || parent->model.lock() != model_ ||
      !parent->Extends(tokens_, image_identity_))
    return full;
  const auto inherited = model_->executor_->SnapshotInheritedBytes(
      *session_, KeptHiddenRows(), parent->executor.get());
  // The full bound includes worst-case descriptor headroom, so subtracting
  // only the inherited region data leaves the new owners (tail buffers,
  // boundary capsule, metadata) covered with margin.
  return full - std::min<std::uint64_t>(inherited, full);
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshot(
    std::string* error_msg) const {
  if (!valid_ || tokens_.empty() || tokens_.size() != session_->position() ||
      tokens_.size() > std::numeric_limits<std::uint32_t>::max()) {
    AssignError(error_msg, "snapshot needs a synced, non-empty context");
    return nullptr;
  }
  const auto token_count = static_cast<std::uint32_t>(tokens_.size());
  const std::uint32_t hidden_rows = KeptHiddenRows();
  const std::uint64_t executor_bytes =
      model_->executor_->SnapshotBytes(*session_, hidden_rows);
  const std::uint64_t host_bytes = SessionSnapshotHostBytes(
      token_count, model_->VocabSize(), image_identity_.size());
  if (!session_->CheckCancellation(error_msg))
    return nullptr;
  auto parent = snapshot_parent_.lock();
  if (parent && (parent->model.lock() != model_ ||
                 !parent->Extends(tokens_, image_identity_)))
    parent.reset();
  auto data = std::make_shared<SessionSnapshotData>();
  data->model = model_;
  const SessionSnapshotHeader header{
      .magic = kSessionSnapshotMagic,
      .version = kSnapshotPayloadVersion,
      .vocab_size = model_->VocabSize(),
      .token_count = token_count,
      .hidden_rows = hidden_rows,
      .executor_bytes = executor_bytes,
      .draft_policy = draft_length_.State(),
      .image_identity_bytes =
          static_cast<std::uint32_t>(image_identity_.size()),
      .policy_concurrency = model_->DecodeConcurrency(),
  };
  data->host = snapshot::Buffer::Capture(host_bytes, [&](auto destination) {
    auto* out = destination.data();
    std::memcpy(out, &header, sizeof(header));
    out += sizeof(header);
    if (!image_identity_.empty())
      std::memcpy(out, image_identity_.data(), image_identity_.size());
    out += image_identity_.size();
    std::memcpy(out, tokens_.data(), tokens_.size() * sizeof(std::int32_t));
    out += tokens_.size() * sizeof(std::int32_t);
    std::memcpy(out, logits_.data(), logits_.size() * sizeof(float));
    return true;
  });
  data->executor = model_->executor_->SaveSharedSnapshot(
      *session_, hidden_rows, parent ? parent->executor.get() : nullptr,
      error_msg);
  if (!data->executor || !session_->CheckCancellation(error_msg))
    return nullptr;
  const auto owners = data->executor->StorageOwners();
  data->owners.reserve(owners.size() + 2);
  data->owners.push_back(snapshot::Buffer::Owner(data->host));
  data->owners.insert(data->owners.end(), owners.begin(), owners.end());
  data->metadata = std::make_shared<const std::uint8_t>(0);
  data->owners.push_back(
      {data->metadata,
       sizeof(SessionSnapshotData) + sizeof(SessionSnapshot) + 1 +
           data->owners.capacity() * sizeof(SnapshotStorageOwner)});
  std::unique_ptr<SessionSnapshot> result(new SessionSnapshot(data));
  snapshot_parent_ = std::move(data);
  return result;
}

bool Session::RestoreSnapshot(const SessionSnapshot& snapshot,
                              std::string* error_msg) {
  if (snapshot.data_->model.lock() != model_) {
    AssignError(error_msg, "session snapshot belongs to another model");
    return false;
  }
  if (!RestoreSnapshotImpl(snapshot.data_->host->bytes(),
                           snapshot.data_->executor.get(), error_msg))
    return false;
  snapshot_parent_ = snapshot.data_;
  return true;
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::string* error_msg) {
  return RestoreSnapshotImpl(payload, nullptr, error_msg);
}

bool Session::RestoreSnapshotImpl(std::span<const std::uint8_t> payload,
                                  const snapshot::Storage* executor,
                                  std::string* error_msg) {
  SessionSnapshotHeader header{};
  if (payload.size() < sizeof(header)) {
    AssignError(error_msg, "session snapshot is truncated");
    return false;
  }
  std::memcpy(&header, payload.data(), sizeof(header));
  if (header.magic != kSessionSnapshotMagic ||
      header.version != kSnapshotPayloadVersion) {
    AssignError(error_msg, "session snapshot format is not supported");
    return false;
  }
  if (header.vocab_size != model_->VocabSize() || header.token_count == 0 ||
      header.policy_concurrency != model_->DecodeConcurrency() ||
      header.token_count > ContextSize() ||
      (header.image_identity_bytes != 0 && header.image_identity_bytes != 32) ||
      (executor != nullptr && header.executor_bytes != executor->size()) ||
      payload.size() != SessionSnapshotHostBytes(header.token_count,
                                                 header.vocab_size,
                                                 header.image_identity_bytes) +
                            (executor ? 0 : header.executor_bytes)) {
    AssignError(error_msg, "session snapshot does not fit this session");
    return false;
  }
  auto restored_policy = draft_length_;
  if (!restored_policy.Restore(header.draft_policy)) {
    AssignError(error_msg, "session snapshot draft policy is invalid");
    return false;
  }
  const std::uint8_t* in = payload.data() + sizeof(header);
  std::vector<std::uint8_t> image_identity(in,
                                           in + header.image_identity_bytes);
  if (!image_identity.empty() && image_identity != image_identity_) {
    AssignError(error_msg,
                "image snapshot requires its matching prompt attachment");
    return false;
  }
  in += header.image_identity_bytes;
  std::vector<std::int32_t> tokens(header.token_count);
  std::memcpy(tokens.data(), in, tokens.size() * sizeof(std::int32_t));
  in += tokens.size() * sizeof(std::int32_t);
  std::vector<float> logits(header.vocab_size);
  std::memcpy(logits.data(), in, logits.size() * sizeof(float));
  in += logits.size() * sizeof(float);

  snapshot_parent_.reset();
  if (image_identity.empty())
    ConfigureVision(nullptr);
  valid_ = false;
  rocm::Executor::SnapshotInfo info;
  const auto remaining = ContextSize() - header.token_count;
  const auto next_drafts =
      MtpEnabled() ? restored_policy.Choose(remaining ? remaining - 1 : 0,
                                            header.token_count)
                   : 0;
  const bool restored =
      executor ? model_->executor_->RestoreSnapshot(*session_, *executor, &info,
                                                    error_msg, next_drafts)
               : model_->executor_->RestoreSnapshot(
                     *session_,
                     {in, static_cast<std::size_t>(header.executor_bytes)},
                     &info, error_msg, next_drafts);
  if (!restored) {
    Reset();
    return false;
  }
  if (info.position != header.token_count ||
      info.hidden_rows != header.hidden_rows) {
    Reset();
    AssignError(error_msg, "session snapshot positions are inconsistent");
    return false;
  }
  image_identity_ = std::move(image_identity);
  tokens_ = std::move(tokens);
  logits_ = std::move(logits);
  hidden_base_ = info.position - info.hidden_rows;
  draft_token_ = 0;
  draft_length_ = restored_policy;
  single_policy_.Reset();
  // tokens_ was replaced wholesale; the lookup watermark is stale.
  lookup_cache_.Reset();
  stats_ = {};
  valid_ = true;
  return true;
}

SessionSnapshot::SessionSnapshot(
    std::shared_ptr<const SessionSnapshotData> data)
    : data_(std::move(data)) {}

std::uint64_t SessionSnapshot::SizeBytes() const noexcept {
  return data_->host->bytes().size() + data_->executor->size();
}

std::span<const SnapshotStorageOwner> SessionSnapshot::StorageOwners()
    const noexcept {
  return data_->owners;
}

void SessionSnapshot::StreamTo(
    const std::function<void(std::span<const std::uint8_t>)>& sink) const {
  sink(data_->host->bytes());
  data_->executor->StreamTo(sink);
}

bool SessionSnapshot::CopyTo(
    std::span<std::uint8_t> destination) const noexcept {
  if (destination.size() != SizeBytes())
    return false;
  const auto host = data_->host->bytes();
  std::memcpy(destination.data(), host.data(), host.size());
  return data_->executor->CopyTo(0, destination.subspan(host.size()));
}

bool Session::DraftReplay(std::int32_t next_token,
                          std::vector<std::int32_t>* replay,
                          std::int32_t* hidden_row,
                          std::string* error_msg) const {
  // The draft block trails the trunk: MTP position i consumes token i+1 and
  // the trunk's hidden of position i, so positions up to the current one are
  // replayed once their successor token is known. The session keeps hidden
  // rows of positions [hidden_base_, tokens_.size()).
  rocm::Executor& exec = *model_->executor_;
  const auto size = static_cast<std::uint32_t>(tokens_.size());
  const std::uint32_t mp = exec.MtpPosition(*session_);
  if (mp >= size) {
    return true;
  }
  if (mp < hidden_base_) {
    AssignError(error_msg, "draft block fell behind the kept hidden rows");
    return false;
  }
  replay->assign(tokens_.begin() + mp + 1, tokens_.end());
  replay->push_back(next_token);
  *hidden_row = static_cast<std::int32_t>(mp - hidden_base_);
  return true;
}

bool Session::DraftCatchUp(std::int32_t next_token, bool propose,
                           std::string* error_msg,
                           MtpCandidateLogits* candidates) {
  std::vector<std::int32_t> replay;
  std::int32_t hidden_row = 0;
  if (!DraftReplay(next_token, &replay, &hidden_row, error_msg))
    return false;
  if (replay.empty())
    return true;
  auto& exec = *model_->executor_;
  if (!exec.MtpForward(
          *session_, replay, hidden_row,
          {.token = propose && candidates == nullptr ? &draft_token_ : nullptr,
           .candidates = candidates},
          error_msg)) {
    return false;
  }
  return true;
}

bool Session::DraftCatchUpBatch(std::span<const AdvanceRequest> requests,
                                std::string* error_msg) {
  if (requests.empty())
    return true;
  auto& exec = *requests.front().session->model_->executor_;
  std::vector<std::vector<std::int32_t>> replays(requests.size());
  std::vector<rocm::Executor::MtpBatchItem> items;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto& session = *requests[i].session;
    if (!session.MtpEnabled())
      continue;
    std::int32_t hidden_row = 0;
    if (!session.DraftReplay(requests[i].token, &replays[i], &hidden_row,
                             error_msg))
      return false;
    if (replays[i].empty())
      continue;
    items.push_back({session.session_.get(), replays[i], hidden_row});
  }
  return items.empty() || exec.MtpForwardBatch(items, error_msg);
}

bool Session::Feed(std::span<const std::int32_t> tokens, std::string* error_msg,
                   bool prefill) {
  rocm::Executor& exec = *model_->executor_;
  for (std::size_t off = 0; off < tokens.size(); off += exec.max_batch()) {
    const std::size_t n =
        std::min<std::size_t>(exec.max_batch(), tokens.size() - off);
    const auto chunk = tokens.subspan(off, n);
    if (MtpEnabled() && !tokens_.empty() &&
        !DraftCatchUp(chunk[0], false, error_msg)) {
      return false;
    }
    const auto mode = prefill ? rocm::Executor::ForwardMode::kPrefill
                              : rocm::Executor::ForwardMode::kDecode;
    if (!exec.Forward(*session_, chunk, 1, logits_.data(), mode, error_msg)) {
      return false;
    }
    hidden_base_ = static_cast<std::uint32_t>(
        tokens_.size() + n - std::min<std::size_t>(n, exec.max_speculative()));
    tokens_.insert(tokens_.end(), chunk.begin(), chunk.end());
  }
  return true;
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  single_policy_.Reset();
  if (prompt.empty()) {
    AssignError(error_msg, "prompt is empty");
    return false;
  }
  if (prompt.size() > ContextSize()) {
    AssignError(error_msg, "prompt exceeds the session context");
    return false;
  }
  if (!valid_)
    Reset();
  // Recurrent state cannot be rewound, so any divergence restarts the
  // session; an extension only feeds the new tail.
  std::size_t common = 0;
  while (common < tokens_.size() && common < prompt.size() &&
         tokens_[common] == prompt[common]) {
    ++common;
  }
  if (common == prompt.size() && common == tokens_.size()) {
    return true;
  }
  if (common != tokens_.size()) {
    Reset();
    common = 0;
  }
  valid_ = false;
  const bool ok = Feed(prompt.subspan(common), error_msg, true);
  valid_ = ok;
  return ok;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  if (tokens_.size() >= ContextSize()) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  valid_ = false;
  const bool ok = Feed(std::span<const std::int32_t>(&token, 1), error_msg);
  valid_ = ok;
  return ok;
}

struct Session::PendingDecode {
  std::vector<std::int32_t> chain;
  std::vector<MtpProposal> proposals;
  std::uint32_t base{0};
  bool speculative{false};
  bool sampled{false};
  bool gpu_greedy{false};
  bool gpu_verification{false};
  std::size_t width{0};
  std::optional<sampling::SamplerState> draft_sampler;
  std::uint64_t draft_rng{0};
  MtpCandidateLogits candidates;
  std::int32_t draft{0};
  /// Batched epilogue: predictions verified once for the whole round, and
  /// a pinned slot holding this session's frontier logits until the batch
  /// epilogue synchronizes.
  const rocm::ArgmaxCandidate* precomputed_greedy{nullptr};
  float* rollback_frontier{nullptr};
  bool defer_rollback_wait{false};
};

void Session::AppendDraft(PendingDecode& pending) {
  if (pending.sampled) {
    pending.proposals.push_back(SampleMtpProposal(
        pending.candidates, *pending.draft_sampler, &pending.draft_rng));
    pending.draft = static_cast<std::int32_t>(pending.proposals.back().token);
    pending.draft_sampler->Accept(pending.proposals.back().token);
  }
  pending.chain.push_back(pending.draft);
}

void Session::ApplyLookupProposals(PendingDecode* pending) {
  if (!model_->options_.prompt_lookup || !pending->speculative ||
      !pending->gpu_greedy || pending->chain.size() < 2) {
    return;
  }
  // Halogen production caps lookup chains at three tokens; deeper slots
  // keep their MTP proposals.
  constexpr std::size_t kLookupChain = 3;
  lookup_cache_.CatchUp(tokens_);
  std::array<std::int32_t, lookup::ContextNgramCache::kMaxContextTokens>
      window{};
  std::size_t window_size =
      std::min<std::size_t>(tokens_.size(), window.size());
  std::copy(tokens_.end() - static_cast<std::ptrdiff_t>(window_size),
            tokens_.end(), window.begin());
  const std::size_t limit =
      std::min<std::size_t>(pending->chain.size() - 1, kLookupChain);
  for (std::size_t i = 0; i < limit; ++i) {
    if (auto proposal = lookup_cache_.ProposeOne(
            std::span(window).first(window_size))) {
      pending->chain[i + 1] = static_cast<std::int32_t>(*proposal);
    }
    // The chain content — substituted or not — is the context the next
    // proposal position sees; the trunk stops at the first rejection
    // either way.
    if (window_size == window.size()) {
      std::move(window.begin() + 1, window.end(), window.begin());
      window.back() = pending->chain[i + 1];
    } else {
      window[window_size++] = pending->chain[i + 1];
    }
  }
}

bool Session::PrepareDecode(const DecodeRequest& request,
                            PendingDecode* pending, std::string* error_msg,
                            bool defer_head,
                            std::optional<std::uint32_t> batch_drafts) {
  const auto max_tokens = request.max_tokens;
  auto& sampler = *request.sampler;
  auto* result = request.result;
  const bool stop_at_eos = request.stop_at_eos;
  if (result == nullptr || max_tokens == 0 || tokens_.empty()) {
    AssignError(
        error_msg,
        "decode needs an output, a positive budget and a synced prompt");
    return false;
  }
  *result = {};
  const auto is_stop = [&](std::int32_t token) {
    return stop_at_eos && model_->IsStopToken(token);
  };
  rocm::Executor& exec = *model_->executor_;
  const std::size_t room = ContextSize() - tokens_.size();
  const std::size_t cap = std::min<std::size_t>(
      {max_tokens, room,
       sampler.has_constraint() ? std::size_t{1} : exec.max_speculative()});
  const std::size_t width =
      MtpEnabled() && cap > 1
          ? 1 + (batch_drafts ? std::min<std::uint32_t>(*batch_drafts, cap - 1)
                              : draft_length_.Choose(
                                    static_cast<std::uint32_t>(cap - 1),
                                    static_cast<std::uint32_t>(tokens_.size())))
          : cap;
  if (width == 0) {
    if (sampler.has_constraint() && !sampler.constraint_complete()) {
      AssignError(error_msg,
                  "context exhausted before constrained output completed");
      return false;
    }
    result->stop = true;
    return true;
  }
  const auto anchor = static_cast<std::int32_t>(sampler.Sample(logits_));
  if (is_stop(anchor)) {
    result->stop = true;
    return true;
  }
  if (!MtpEnabled() || width < 2) {
    if (MtpEnabled() && !defer_head &&
        !DraftCatchUp(anchor, false, error_msg)) {
      return false;
    }
    pending->chain = {anchor};
    pending->base = static_cast<std::uint32_t>(tokens_.size());
    return true;
  }

  const std::uint32_t base = static_cast<std::uint32_t>(tokens_.size());
  const bool sampled = sampler.config().uses_random_sampling();
  const bool gpu_greedy =
      !sampler.has_constraint() && sampler.config().can_use_unmodified_argmax();
  const bool gpu_verification = gpu_greedy;
  if (!defer_head && !DraftCatchUp(anchor, true, error_msg,
                                   sampled ? &pending->candidates : nullptr)) {
    return false;
  }
  // A cycle-local proposal stream needs no pending RNG state in snapshots.
  // Target verification keeps its own draws after this independent seed.
  pending->draft_rng =
      sampled ? sampling::NextRandom(sampler.mutable_rng_state()) : 0;
  pending->draft_sampler = sampler;
  pending->draft_sampler->Accept(static_cast<sampling::TokenId>(anchor));
  pending->chain = {anchor};
  pending->draft = draft_token_;
  pending->width = width;
  pending->base = base;
  pending->speculative = true;
  pending->sampled = sampled;
  pending->gpu_greedy = gpu_greedy;
  pending->gpu_verification = gpu_verification;
  if (!gpu_verification && verify_logits_.empty()) {
    verify_logits_.resize(exec.max_speculative() * model_->VocabSize());
  }
  if (!defer_head) {
    while (pending->chain.size() < width) {
      AppendDraft(*pending);
      if (pending->chain.size() < width &&
          !exec.MtpForward(
              *session_, std::span<const std::int32_t>(&pending->draft, 1), -1,
              {.token = sampled ? nullptr : &pending->draft,
               .candidates = sampled ? &pending->candidates : nullptr},
              error_msg)) {
        return false;
      }
    }
    ApplyLookupProposals(pending);
  }
  return true;
}

bool Session::FinishDecode(const DecodeRequest& request,
                           const PendingDecode& pending,
                           std::string* error_msg) {
  auto& sampler = *request.sampler;
  auto* result = request.result;
  auto& exec = *model_->executor_;
  const auto& chain = pending.chain;
  const auto& proposals = pending.proposals;
  const auto base = pending.base;
  const bool sampled = pending.sampled;
  const bool gpu_greedy = pending.gpu_greedy;
  const bool gpu_verification = pending.gpu_verification;
  const auto anchor = chain.front();
  const auto k = static_cast<std::uint32_t>(chain.size());
  const auto vocab = model_->VocabSize();
  const auto is_stop = [&](std::int32_t token) {
    return request.stop_at_eos && model_->IsStopToken(token);
  };
  if (!pending.speculative) {
    draft_length_.ObserveArToken();
    hidden_base_ = base;
    tokens_.push_back(anchor);
    sampler.Accept(static_cast<sampling::TokenId>(anchor));
    result->tokens.push_back(anchor);
    return true;
  }
  sampler.Accept(static_cast<sampling::TokenId>(anchor));
  std::array<rocm::ArgmaxCandidate, kMaxMtpDraftTokens> greedy{};
  if (gpu_greedy) {
    if (pending.precomputed_greedy != nullptr) {
      std::copy_n(pending.precomputed_greedy, k - 1, greedy.begin());
    } else if (!exec.GreedyMtpPredictions(std::span(greedy).first(k - 1),
                                          error_msg)) {
      return false;
    }
  }
  std::uint32_t keep = 1;
  std::optional<std::int32_t> correction;
  while (keep < k) {
    if (gpu_greedy) {
      const auto& prediction = greedy[keep - 1];
      if (!std::isfinite(prediction.value)) {
        AssignError(error_msg, "logit distribution contains no finite values");
        return false;
      }
      if (is_stop(prediction.index)) {
        result->stop = true;
        break;
      }
      if (prediction.index != chain[keep]) {
        break;
      }
      sampler.Accept(static_cast<sampling::TokenId>(prediction.index));
      ++keep;
      continue;
    }
    if (sampled) {
      const auto verified =
          VerifyMtpProposal(std::span<const float>(verify_logits_)
                                .subspan((keep - 1) * vocab, vocab),
                            proposals[keep - 1], sampler);
      const auto token = static_cast<std::int32_t>(verified.token);
      const bool accepted = verified.accepted;
      if (is_stop(token)) {
        result->stop = true;
        break;
      }
      if (!accepted) {
        correction = token;
        break;
      }
      sampler.Accept(static_cast<sampling::TokenId>(token));
      ++keep;
      continue;
    }
    const auto decision = VerifyDraft(std::span<const float>(verify_logits_)
                                          .subspan((keep - 1) * vocab, vocab),
                                      chain[keep], sampler, is_stop);
    if (decision != DraftDecision::kAccept) {
      result->stop = decision == DraftDecision::kStop;
      break;
    }
    ++keep;
  }
  if (!exec.Rollback(*session_, keep, error_msg,
                     gpu_verification ? (pending.rollback_frontier
                                            ? pending.rollback_frontier
                                            : logits_.data())
                                      : nullptr,
                     !pending.defer_rollback_wait)) {
    return false;
  }
  if (!gpu_verification) {
    std::copy_n(verify_logits_.data() + (keep - 1) * vocab, vocab,
                logits_.begin());
  }
  hidden_base_ = base;
  tokens_.insert(tokens_.end(), chain.begin(), chain.begin() + keep);
  result->tokens.assign(chain.begin(), chain.begin() + keep);
  stats_.cycles += 1;
  stats_.drafted += k - 1;
  stats_.accepted += keep - 1;
  // A target stop ends the request; it does not classify the remaining
  // proposals as failed predictions.
  draft_length_.Observe(keep - 1, result->stop ? keep - 1 : k - 1, base);

  // The next call knows the next sampled anchor. Defer draft catch-up until
  // then, retaining this session's target hidden rows across interleaving.
  exec.MtpRewind(*session_, base);
  if (correction) {
    // Evaluate the residual as the next cycle's anchor, avoiding a separate
    // target pass. Preserve the actual draw: resampling p would be biased.
    sampler.DeferSample(static_cast<sampling::TokenId>(*correction));
  }
  return true;
}

bool Session::DecodeStep(std::size_t max_tokens,
                         sampling::SamplerState& sampler, DecodeResult* result,
                         std::string* error_msg, bool stop_at_eos) {
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  valid_ = false;
  const DecodeRequest request{this, max_tokens, &sampler, result, stop_at_eos};
  std::optional<std::uint32_t> measured_drafts;
  std::chrono::steady_clock::time_point cycle_start;
  const auto context = Position();
  const auto cap =
      std::min<std::size_t>({max_tokens, ContextSize() - tokens_.size(),
                             model_->executor_->max_speculative()});
  if (MtpEnabled() && cap > 1 && !sampler.has_constraint() &&
      !sampler.config().uses_random_sampling()) {
    const MtpBatchController::Row row{&draft_length_,
                                      static_cast<std::uint32_t>(cap - 1)};
    measured_drafts = single_policy_.Choose(std::span(&row, 1), context);
    cycle_start = std::chrono::steady_clock::now();
  }
  PendingDecode pending;
  if (!PrepareDecode(request, &pending, error_msg, false, measured_drafts)) {
    return false;
  }
  if (pending.chain.empty()) {
    valid_ = true;
    return true;
  }
  float* logits = !pending.speculative       ? logits_.data()
                  : pending.gpu_verification ? nullptr
                                             : verify_logits_.data();
  if (!model_->executor_->Forward(
          *session_, pending.chain, pending.chain.size(), logits,
          pending.speculative ? rocm::Executor::ForwardMode::kVerify
                              : rocm::Executor::ForwardMode::kDecode,
          error_msg)) {
    return false;
  }
  const bool ok = FinishDecode(request, pending, error_msg);
  if (ok && measured_drafts && !result->stop) {
    const auto ms = std::chrono::duration<float, std::milli>(
                        std::chrono::steady_clock::now() - cycle_start)
                        .count();
    single_policy_.Observe(1, context, *measured_drafts, ms);
  }
  valid_ = ok;
  return ok;
}

template<class Request>
bool Session::RunIsolatedBatch(std::span<const Request> requests,
                               std::string* error_msg) {
  if (requests.empty() || requests.size() > 8) {
    AssignError(error_msg, "batch must contain 1..8 sessions");
    return false;
  }
  std::array<BatchOutcome, 8> outcomes{};
  std::array<std::uint64_t, 8> epochs{};
  std::array<sampling::SamplerState::DrawState, 8> draws{};
  std::vector<Request> active;
  active.reserve(requests.size());
  Model* model = nullptr;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    auto& outcome = outcomes[i];
    if (r.session)
      epochs[i] = r.session->session_->MutationEpoch();
    bool valid = r.session && r.session->valid_;
    if constexpr (std::is_same_v<Request, DecodeRequest>) {
      valid = valid && r.sampler && r.result && r.max_tokens > 0 &&
              !r.session->tokens_.empty();
    } else {
      valid = valid && r.token >= 0 &&
              static_cast<std::uint32_t>(r.token) <
                  r.session->model_->VocabSize() &&
              r.session->Position() < r.session->ContextSize();
    }
    for (std::size_t j = 0; j < requests.size(); ++j) {
      if (i == j)
        continue;
      valid = valid && r.session != requests[j].session;
      if constexpr (std::is_same_v<Request, DecodeRequest>)
        valid = valid && r.sampler != requests[j].sampler &&
                r.result != requests[j].result;
    }
    if (valid && model && r.session->model_.get() != model)
      valid = false;
    if (!valid) {
      outcome.error = "invalid or non-independent batch request";
      continue;
    }
    if (!r.session->session_->CheckCancellation(&outcome.error))
      continue;
    model = r.session->model_.get();
    if constexpr (std::is_same_v<Request, DecodeRequest>) {
      draws[i] = r.sampler->SaveDrawState();
    }
    auto copy = r;
    copy.outcome = &outcome;
    active.push_back(copy);
  }
  std::string shared_error;
  try {
    if (!active.empty()) {
      if constexpr (std::is_same_v<Request, DecodeRequest>)
        (void)DecodeBatchImpl(active, &shared_error);
      else
        (void)EvaluateBatchImpl(active, &shared_error);
    }
  } catch (const std::exception& exception) {
    shared_error = exception.what();
  }
  bool success = true;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    auto& outcome = outcomes[i];
    if (!outcome.completed && outcome.error.empty()) {
      auto& session = *r.session;
      if (session.session_->Cancelled()) {
        outcome.error = "generation cancelled";
      } else if (session.session_->MutationEpoch() == epochs[i]) {
        // A preparation/batch-allocation failure did not touch this peer.
        // Retry it independently, without changing its cached frontier.
        session.valid_ = true;
        if constexpr (std::is_same_v<Request, DecodeRequest>) {
          r.sampler->RestoreDrawState(draws[i]);
        }
        try {
          if constexpr (std::is_same_v<Request, DecodeRequest>)
            outcome.completed =
                session.DecodeStep(r.max_tokens, *r.sampler, r.result,
                                   &outcome.error, r.stop_at_eos);
          else
            outcome.completed = session.Evaluate(r.token, &outcome.error);
        } catch (const std::exception& exception) {
          outcome.error = exception.what();
        }
      } else {
        outcome.error =
            shared_error.empty() ? "batch execution failed" : shared_error;
      }
    }
    if (!outcome.completed) {
      if (outcome.error.empty())
        outcome.error = "batch request failed";
      if (r.session && r.session->session_->MutationEpoch() != epochs[i])
        r.session->valid_ = false;
      if (success)
        AssignError(error_msg, outcome.error);
      success = false;
    } else {
      r.session->valid_ = true;
    }
    if (r.outcome)
      *r.outcome = std::move(outcome);
  }
  return success;
}

bool Session::DecodeBatch(std::span<const DecodeRequest> requests,
                          std::string* error_msg) {
  return RunIsolatedBatch(requests, error_msg);
}

bool Session::EvaluateBatch(std::span<const AdvanceRequest> requests,
                            std::string* error_msg) {
  return RunIsolatedBatch(requests, error_msg);
}

bool Session::DecodeBatchImpl(std::span<const DecodeRequest> requests,
                              std::string* error_msg) {
  if (requests.size() == 1) {
    const auto& r = requests.front();
    r.outcome->completed = r.session->DecodeStep(
        r.max_tokens, *r.sampler, r.result, &r.outcome->error, r.stop_at_eos);
    return r.outcome->completed;
  }
  auto& exec = *requests.front().session->model_->executor_;
  std::optional<std::uint32_t> batch_drafts;
  std::uint32_t batch_context = 0;
  auto& policy = requests.front().session->model_->batch_policy_;
  if (std::ranges::all_of(
          requests, [](const auto& r) { return r.session->MtpEnabled(); }) &&
      std::ranges::none_of(requests, [](const auto& request) {
        return request.sampler->has_constraint() ||
               request.sampler->config().uses_random_sampling();
      })) {
    std::array<MtpBatchController::Row, 8> rows{};
    for (std::size_t i = 0; i < requests.size(); ++i) {
      const auto& r = requests[i];
      const auto cap = std::min<std::size_t>(
          r.max_tokens, r.session->ContextSize() - r.session->Position());
      rows[i] = {&r.session->draft_length_,
                 static_cast<std::uint32_t>(
                     std::min<std::size_t>(cap, exec.max_speculative())) -
                     (cap != 0)};
      batch_context = std::max(batch_context, r.session->Position());
    }
    batch_drafts =
        policy.Choose(std::span(rows).first(requests.size()), batch_context);
  }
  const auto cycle_start = std::chrono::steady_clock::now();
  std::vector<PendingDecode> pending(requests.size());
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    try {
      if (!r.session->PrepareDecode(r, &pending[i], &r.outcome->error, true,
                                    batch_drafts))
        pending[i] = {};
    } catch (const std::exception& exception) {
      r.outcome->error = exception.what();
      pending[i] = {};
    }
  }
  std::vector<AdvanceRequest> catchup;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    if (!pending[i].chain.empty())
      catchup.push_back({requests[i].session, pending[i].chain.front()});
  }
  if (!DraftCatchUpBatch(catchup, error_msg))
    return false;
  // Each round shares predictor projections across ready sessions.
  // Attention state, proposal distributions and RNG streams stay private.
  for (;;) {
    std::vector<rocm::Executor::MtpHeadItem> heads;
    std::vector<rocm::Executor::MtpBatchItem> bodies;
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto& p = pending[i];
      if (!requests[i].session->session_->Cancelled() && p.speculative &&
          p.chain.size() < p.width) {
        heads.push_back({requests[i].session->session_.get(),
                         {.token = p.sampled ? nullptr : &p.draft,
                          .candidates = p.sampled ? &p.candidates : nullptr}});
      }
    }
    if (heads.empty())
      break;
    if (!exec.MtpHeads(heads, error_msg))
      return false;
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto& p = pending[i];
      if (requests[i].session->session_->Cancelled() || !p.speculative ||
          p.chain.size() >= p.width)
        continue;
      AppendDraft(p);
      if (p.chain.size() < p.width)
        bodies.push_back({requests[i].session->session_.get(),
                          std::span<const std::int32_t>(&p.draft, 1), -1});
    }
    if (!bodies.empty() && !exec.MtpForwardBatch(bodies, error_msg))
      return false;
  }
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto& p = pending[i];
    if (!requests[i].session->session_->Cancelled() && !p.chain.empty()) {
      requests[i].session->ApplyLookupProposals(&p);
    }
  }
  std::vector<rocm::Executor::BatchItem> items;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    if (!r.session->session_->Cancelled() && !pending[i].chain.empty()) {
      items.push_back({r.session->session_.get(), pending[i].chain,
                       pending[i].speculative});
    }
  }
  if (!items.empty() && !exec.ForwardBatch(items, error_msg))
    return false;
  // Batched greedy epilogue: one argmax over the round's rows, per-session
  // frontier downloads into pinned slots, and a single stream sync.
  // Requires every batched session to verify greedily; otherwise the
  // per-session path below handles each request.
  bool batched_epilogue = !items.empty();
  std::vector<rocm::ArgmaxCandidate> round_predictions;
  std::uint32_t round_rows = 0;
  std::vector<std::pair<std::size_t, std::uint32_t>> deferred;
  std::string epilogue_error;
  std::uint32_t offset = 0;
  if (batched_epilogue) {
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto& p = pending[i];
      const bool included =
          !requests[i].session->session_->Cancelled() && !p.chain.empty();
      if (!included)
        continue;
      if (!p.speculative || !p.gpu_verification) {
        batched_epilogue = false;
        break;
      }
      deferred.emplace_back(i, round_rows);
      round_rows += static_cast<std::uint32_t>(p.chain.size());
      offset = round_rows;
    }
  }
  if (batched_epilogue) {
    round_predictions.resize(round_rows);
    if (!exec.SelectBatchLogits(0, round_rows, nullptr, &epilogue_error) ||
        !exec.GreedyMtpPredictions(round_predictions, &epilogue_error)) {
      batched_epilogue = false;
      epilogue_error.clear();
      round_predictions.clear();
      deferred.clear();
    }
  }
  if (batched_epilogue) {
    for (std::size_t slot = 0; slot < deferred.size(); ++slot) {
      auto& p = pending[deferred[slot].first];
      p.precomputed_greedy = round_predictions.data() + deferred[slot].second;
      p.rollback_frontier =
          exec.BatchFrontierStaging(static_cast<std::uint32_t>(slot),
                                    &epilogue_error);
      if (p.rollback_frontier == nullptr) {
        for (auto& entry : deferred) {
          pending[entry.first].precomputed_greedy = nullptr;
          pending[entry.first].rollback_frontier = nullptr;
          pending[entry.first].defer_rollback_wait = false;
        }
        batched_epilogue = false;
        epilogue_error.clear();
        deferred.clear();
        break;
      }
      p.defer_rollback_wait = true;
    }
  }
  offset = 0;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& p = pending[i];
    const auto& r = requests[i];
    auto& session = *r.session;
    const auto row_offset = offset;
    if (!r.outcome->error.empty()) {
      if (!r.session->session_->Cancelled() && !p.chain.empty())
        offset += p.chain.size();
      continue;
    }
    if (session.session_->Cancelled()) {
      r.outcome->error = "generation cancelled";
      offset += p.chain.size();
      continue;
    }
    if (p.chain.empty()) {
      r.outcome->completed = true;
      continue;
    }
    offset += p.chain.size();
    float* logits = !p.speculative       ? session.logits_.data()
                    : p.gpu_verification ? nullptr
                                         : session.verify_logits_.data();
    try {
      r.outcome->completed =
          (batched_epilogue && p.speculative
               ? true
               : exec.SelectBatchLogits(row_offset, p.chain.size(), logits,
                                        &r.outcome->error)) &&
          session.FinishDecode(r, p, &r.outcome->error);
    } catch (const std::exception& exception) {
      r.outcome->error = exception.what();
    }
  }
  if (!deferred.empty()) {
    // One sync covers every deferred frontier download; the host copies
    // then land in each session's logits for the next round.
    std::string sync_error;
    if (!gufo::hip::WaitStream(exec.stream(), "batched decode epilogue",
                               &sync_error)) {
      for (const auto& [index, slot_offset] : deferred) {
        requests[index].outcome->error = sync_error;
        requests[index].outcome->completed = false;
      }
    } else {
      const auto vocab = requests.front().session->model_->VocabSize();
      for (const auto& [index, slot_offset] : deferred) {
        auto& session = *requests[index].session;
        std::copy_n(pending[index].rollback_frontier, vocab,
                    session.logits_.data());
      }
    }
  }
  if (batch_drafts && items.size() == requests.size() &&
      std::ranges::all_of(requests,
                          [](const auto& r) { return r.outcome->completed; })) {
    const auto ms = std::chrono::duration<float, std::milli>(
                        std::chrono::steady_clock::now() - cycle_start)
                        .count();
    policy.Observe(requests.size(), batch_context, *batch_drafts, ms);
  }
  return true;
}

bool Session::EvaluateBatchImpl(std::span<const AdvanceRequest> requests,
                                std::string* error_msg) {
  if (requests.size() == 1) {
    const auto& r = requests.front();
    r.outcome->completed = r.session->Evaluate(r.token, &r.outcome->error);
    return r.outcome->completed;
  }
  auto& exec = *requests.front().session->model_->executor_;
  if (!DraftCatchUpBatch(requests, error_msg))
    return false;
  std::vector<rocm::Executor::BatchItem> items;
  for (const auto& r : requests) {
    items.push_back({r.session->session_.get(), {&r.token, 1}, false});
  }
  if (!exec.ForwardBatch(items, error_msg)) {
    return false;
  }
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto& session = *requests[i].session;
    auto& outcome = *requests[i].outcome;
    if (session.session_->Cancelled()) {
      outcome.error = "generation cancelled";
      continue;
    }
    try {
      if (!exec.SelectBatchLogits(i, 1, session.logits_.data(), &outcome.error))
        continue;
      session.hidden_base_ = static_cast<std::uint32_t>(session.tokens_.size());
      session.tokens_.push_back(requests[i].token);
      outcome.completed = true;
    } catch (const std::exception& exception) {
      outcome.error = exception.what();
    }
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next
