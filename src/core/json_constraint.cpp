#include "src/core/json_constraint.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gufo {
namespace {

using Id = std::uint32_t;
using Value = json::Value;
constexpr std::size_t kInfinity = std::numeric_limits<std::size_t>::max() / 4;
constexpr unsigned kNull = 1, kBool = 2, kInteger = 4, kReal = 8, kString = 16,
                   kArray = 32, kObject = 64, kAll = 127;

[[noreturn]] void Fail(const std::string& path, const std::string& message) {
  throw std::invalid_argument("JSON constraint " + path + ": " + message);
}

std::string Pointer(const std::string& path, std::string_view key) {
  std::string result = path + '/';
  for (char c : key) {
    if (c == '~')
      result += "~0";
    else if (c == '/')
      result += "~1";
    else
      result += c;
  }
  return result;
}

std::size_t Add(std::size_t a, std::size_t b) {
  return a >= kInfinity || b >= kInfinity || a > kInfinity - b ? kInfinity
                                                               : a + b;
}

std::size_t Multiply(std::size_t a, std::size_t b) {
  if (a == 0 || b == 0)
    return 0;
  return a >= kInfinity / b ? kInfinity : a * b;
}

// Validate scalar UTF-8 without depending on a locale or accepting CESU-8.
std::optional<std::size_t> ScalarLength(std::string_view s) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < s.size(); ++count) {
    const auto c = static_cast<unsigned char>(s[i++]);
    if (c < 0x80)
      continue;
    unsigned length = c >= 0xC2 && c <= 0xDF   ? 2
                      : c >= 0xE0 && c <= 0xEF ? 3
                      : c >= 0xF0 && c <= 0xF4 ? 4
                                               : 0;
    if (length == 0 || s.size() - i < length - 1)
      return std::nullopt;
    for (unsigned j = 1; j < length; ++j) {
      const auto next = static_cast<unsigned char>(s[i++]);
      if (next < 0x80 || next > 0xBF ||
          (j == 1 &&
           ((c == 0xE0 && next < 0xA0) || (c == 0xED && next > 0x9F) ||
            (c == 0xF0 && next < 0x90) || (c == 0xF4 && next > 0x8F))))
        return std::nullopt;
    }
  }
  return count;
}

unsigned TypeOf(const Value& v) {
  if (v.is_null())
    return kNull;
  if (v.is_bool())
    return kBool;
  if (v.is_number())
    return std::floor(v.as_double()) == v.as_double() ? kInteger : kReal;
  if (v.is_string())
    return kString;
  return v.is_array() ? kArray : kObject;
}

struct ValidationLimit {};

// JSON equality, unlike serialized equality, ignores object member order.
// NOLINTNEXTLINE(misc-no-recursion)
bool Equal(const Value& a, const Value& b, std::size_t& work,
           std::size_t call_depth = 0) {
  if (work == 0 || call_depth >= 256)
    throw ValidationLimit{};
  --work;
  if (a.type() != b.type())
    return false;
  if (a.is_null())
    return true;
  if (a.is_bool())
    return a.as_bool() == b.as_bool();
  if (a.is_number())
    return a.number_is_exact() && b.number_is_exact() &&
           a.as_double() == b.as_double();
  if (a.is_string())
    return a.str() == b.str();
  if (a.size() != b.size())
    return false;
  if (a.is_array()) {
    for (std::size_t i = 0; i < a.size(); ++i)
      if (!Equal(a.items()[i], b.items()[i], work, call_depth + 1))
        return false;
  } else {
    for (const auto& [key, value] : a.members()) {
      const auto* other = b.find(key);
      if (!other || !Equal(value, *other, work, call_depth + 1))
        return false;
    }
  }
  return true;
}

bool Annotation(std::string_view key) {
  return key == "title" || key == "description" || key == "default" ||
         key == "examples" || key == "deprecated" || key == "readOnly" ||
         key == "writeOnly" || key == "$comment" || key == "$schema";
}

struct Property {
  std::string name;
  Id schema = 0;
  bool required = false;
};

struct Schema {
  enum class Kind { kAny, kFalse, kRules, kRef, kUnion };
  Kind kind = Kind::kRules;
  std::string path;
  unsigned types = kAll;
  Id ref = 0;
  std::vector<Id> branches;
  bool one_of = false;
  std::vector<Property> properties;
  Id additional = 0;
  Id items = 0;
  std::size_t min_items = 0, max_items = kInfinity;
  std::size_t min_length = 0, max_length = kInfinity;
  bool finite = false;
  std::vector<Value> values;
};

struct Expression {
  enum class Kind {
    kFail,
    kEmpty,
    kLiteral,
    kSequence,
    kAlternative,
    kRepeat,
    kString,
    kNumber,
    kWhitespace
  };
  Kind kind = Kind::kEmpty;
  std::string literal;
  std::vector<Id> children;
  std::size_t minimum = 0;
  std::size_t low = 0, high = 0;
  bool integer = false;
};

struct Program {
  JsonConstraint::Limits limits;
  std::vector<Schema> schemas;
  std::vector<Expression> expressions;
  Id schema_root = 0, root = 0;

  // Runtime validation has its own work/depth bound, including reference hops.
  // NOLINTNEXTLINE(misc-no-recursion)
  bool Matches(Id id, const Value& v, std::size_t depth, std::size_t& work,
               bool skip_finite = false, std::size_t call_depth = 0) const {
    if (work == 0 || call_depth >= 256)
      throw ValidationLimit{};
    if (depth > limits.max_depth)
      return false;
    --work;
    const auto& s = schemas[id];
    if (s.kind == Schema::Kind::kFalse)
      return false;
    if (s.kind == Schema::Kind::kRef)
      return Matches(s.ref, v, depth, work, false, call_depth + 1);
    if (s.kind == Schema::Kind::kUnion) {
      unsigned matches = 0;
      for (Id branch : s.branches) {
        if (Matches(branch, v, depth, work, false, call_depth + 1)) {
          if (!s.one_of)
            return true;
          if (++matches > 1)
            return false;
        }
      }
      return matches == 1;
    }
    if ((s.types & TypeOf(v)) == 0)
      return false;
    if (s.finite && !skip_finite &&
        std::none_of(s.values.begin(), s.values.end(), [&](const Value& entry) {
          return Equal(v, entry, work, call_depth + 1);
        }))
      return false;
    if (v.is_number())
      return std::isfinite(v.as_double()) &&
             ((s.types & kReal) != 0 || v.number_is_exact());
    if (v.is_string()) {
      const auto length = ScalarLength(v.str());
      return length && *length >= s.min_length && *length <= s.max_length;
    }
    if (v.is_array()) {
      if (depth == limits.max_depth || v.size() < s.min_items ||
          v.size() > s.max_items)
        return false;
      for (const auto& item : v.items())
        if (!Matches(s.items, item, depth + 1, work, false, call_depth + 1))
          return false;
    }
    if (v.is_object()) {
      if (depth == limits.max_depth)
        return false;
      std::unordered_set<std::string_view> keys;
      for (const auto& p : s.properties)
        if (p.required && !v.contains(p.name))
          return false;
      for (const auto& [key, value] : v.members()) {
        if (!ScalarLength(key) || !keys.insert(key).second)
          return false;
        const auto it =
            std::find_if(s.properties.begin(), s.properties.end(),
                         [&](const Property& p) { return p.name == key; });
        if (!Matches(it == s.properties.end() ? s.additional : it->schema,
                     value, depth + 1, work, false, call_depth + 1))
          return false;
      }
    }
    return true;
  }

  bool Validate(Id id, const Value& v, bool skip_finite = false) const {
    // Exhaustion is indeterminate, never a branch-local mismatch: treating it
    // as false could turn two matching oneOf branches into exactly one.
    std::size_t work = 1048576;
    return Matches(id, v, 0, work, skip_finite);
  }
};

class Compiler {
public:
  Compiler(const Value& schema, const JsonConstraint::Limits& limits)
      : source_(schema) {
    program_ = std::make_shared<Program>();
    program_->limits = limits;
    CheckLimits();
    work_ = limits.max_compile_steps;
    std::size_t nodes = 0, bytes = 0;
    CheckAst(source_, "#", 0, nodes, bytes, false);
    Schema any;
    any.kind = Schema::Kind::kAny;
    any.path = "#";
    Schema never;
    never.kind = Schema::Kind::kFalse;
    never.path = "#";
    program_->schemas = {any, never};
  }

  std::shared_ptr<const Program> Compile() {
    program_->schema_root = Parse(source_, "#");
    std::vector<unsigned char> marks(program_->schemas.size());
    ref_heights_.resize(program_->schemas.size());
    for (Id i = 0; i < program_->schemas.size(); ++i)
      (void)CheckUnguarded(i, marks);
    for (Id i = 0; i < program_->schemas.size(); ++i) {
      const auto& s = program_->schemas[i];
      if (!s.one_of)
        continue;
      for (std::size_t a = 0; a < s.branches.size(); ++a)
        for (std::size_t b = a + 1; b < s.branches.size(); ++b)
          if (!Disjoint(s.branches[a], s.branches[b], 0))
            Fail(s.path + "/oneOf",
                 "branches " + std::to_string(a) + " and " + std::to_string(b) +
                     " overlap or cannot be proven disjoint; use anyOf");
    }
    Expression fail;
    fail.kind = Expression::Kind::kFail;
    fail.minimum = kInfinity;
    program_->expressions.push_back(fail);
    program_->expressions.emplace_back();
    Expression ws;
    ws.kind = Expression::Kind::kWhitespace;
    ws.high = program_->limits.max_whitespace;
    whitespace_ = Make(std::move(ws));
    const Id body = Build(program_->schema_root, 0);
    if (body == 0)
      Fail("#", "schema has no finite completion within configured limits");
    program_->root = Sequence({whitespace_, body, whitespace_});
    if (program_->expressions[program_->root].minimum >
        program_->limits.max_output_bytes)
      Fail("#", "minimum output exceeds max_output_bytes");
    return program_;
  }

private:
  const Value& source_;
  std::shared_ptr<Program> program_;
  std::unordered_map<const Value*, Id> parsed_;
  std::map<std::pair<Id, std::size_t>, Id> built_;
  mutable std::unordered_map<Id, unsigned> type_masks_;
  std::vector<std::size_t> ref_heights_;
  mutable std::size_t work_ = 0;
  mutable std::size_t call_depth_ = 0;
  std::size_t parse_depth_ = 0;
  Id whitespace_ = 0;

  void Step(const std::string& path) const {
    if (work_ == 0)
      Fail(path, "schema compilation work limit exceeded");
    --work_;
  }

  bool ProofMatches(Id id, const Value& value, bool skip_finite = false) const {
    return program_->Matches(id, value, 0, work_, skip_finite, call_depth_);
  }

  struct CallDepth {
    std::size_t& depth;
    CallDepth(std::size_t& value, const std::string& path) : depth(value) {
      if (depth >= 256)
        Fail(path, "combined schema traversal depth limit exceeded (256)");
      ++depth;
    }
    ~CallDepth() { --depth; }
  };

  struct ParseDepth {
    std::size_t& depth;
    explicit ParseDepth(std::size_t& value) : depth(value) { ++depth; }
    ~ParseDepth() { --depth; }
  };

  void CheckLimits() const {
    const auto& l = program_->limits;
    if (l.max_schema_bytes == 0 || l.max_schema_bytes > 4194304 ||
        l.max_schema_nodes < 2 || l.max_schema_nodes > 65536 ||
        l.max_grammar_nodes < 16 || l.max_grammar_nodes > 1048576 ||
        l.max_compile_steps == 0 || l.max_compile_steps > 16777216 ||
        l.max_states == 0 || l.max_states > 65536 || l.max_depth == 0 ||
        l.max_depth > 128 || l.max_properties > 4096 || l.max_items > 65536 ||
        l.max_string_length > 1048576 || l.max_number_bytes < 1 ||
        l.max_number_bytes > 4096 || l.max_whitespace > 1024 ||
        l.max_output_bytes == 0 || l.max_output_bytes > 16777216)
      Fail("#", "invalid safety limits");
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  void CheckAst(const Value& v, const std::string& path, std::size_t depth,
                std::size_t& nodes, std::size_t& bytes, bool literal) const {
    CallDepth call(call_depth_, path);
    Step(path);
    const auto& l = program_->limits;
    if (++nodes > l.max_schema_nodes || depth > 128)
      Fail(path, "schema node/depth limit exceeded");
    bytes = Add(bytes, 4);
    if (v.is_number()) {
      if (!std::isfinite(v.as_double()))
        Fail(path, "non-finite number");
      if (literal && (!v.number_is_exact() ||
                      std::abs(v.as_double()) > 9007199254740991.0))
        Fail(path,
             "numeric const/enum loses precision or exceeds exact binary64 "
             "integer range");
      bytes = Add(bytes, v.dump().size());
    }
    if (v.is_string()) {
      if (!ScalarLength(v.str()))
        Fail(path, "invalid UTF-8");
      bytes = Add(bytes, v.str().size());
    }
    if (bytes > l.max_schema_bytes)
      Fail(path, "schema byte limit exceeded");
    if (v.is_array()) {
      for (std::size_t i = 0; i < v.size(); ++i)
        CheckAst(v.items()[i], Pointer(path, std::to_string(i)), depth + 1,
                 nodes, bytes, literal);
    }
    if (v.is_object()) {
      std::unordered_set<std::string_view> keys;
      for (const auto& [key, child] : v.members()) {
        if (!ScalarLength(key) || !keys.insert(key).second)
          Fail(Pointer(path, key), "invalid UTF-8 or duplicate key");
        bytes = Add(bytes, key.size());
        CheckAst(child, Pointer(path, key), depth + 1, nodes, bytes, literal);
      }
    }
  }

  std::size_t Bound(const Value& v, const std::string& path,
                    std::size_t maximum) const {
    if (!v.is_number() || !v.number_is_exact() || v.as_double() < 0 ||
        std::floor(v.as_double()) != v.as_double() ||
        v.as_double() > static_cast<double>(maximum))
      Fail(path, "expected nonnegative integer <= " + std::to_string(maximum));
    return static_cast<std::size_t>(v.as_double());
  }

  const Value& Resolve(const Value& ref, const std::string& path) const {
    if (!ref.is_string() || ref.str().empty() || ref.str()[0] != '#')
      Fail(path, "only local JSON Pointer $ref is supported");
    const auto& text = ref.str();
    if (text == "#")
      return source_;
    if (text.size() < 2 || text[1] != '/' ||
        text.find('%') != std::string::npos)
      Fail(path, "expected #/ JSON Pointer (URI percent escapes unsupported)");
    const Value* current = &source_;
    std::size_t start = 2;
    while (true) {
      const auto end = text.find('/', start);
      const auto part = text.substr(
          start, end == std::string::npos ? std::string::npos : end - start);
      std::string key;
      for (std::size_t i = 0; i < part.size(); ++i) {
        if (part[i] != '~') {
          key += part[i];
        } else {
          if (++i == part.size() || (part[i] != '0' && part[i] != '1'))
            Fail(path, "invalid JSON Pointer escape");
          key += part[i] == '0' ? '~' : '/';
        }
      }
      if (current->is_object()) {
        current = current->find(key);
      } else if (current->is_array()) {
        if (key.empty() || (key.size() > 1 && key[0] == '0') ||
            key.find_first_not_of("0123456789") != std::string::npos ||
            key.size() > 8)
          Fail(path, "invalid JSON Pointer array index");
        const auto index = static_cast<std::size_t>(std::stoul(key));
        current = index < current->size() ? &current->items()[index] : nullptr;
      } else {
        current = nullptr;
      }
      if (!current)
        Fail(path, "unresolved local reference " + text);
      if (end == std::string::npos)
        return *current;
      start = end + 1;
    }
  }

  unsigned ParseType(const Value& v, const std::string& path) const {
    if (!v.is_string())
      Fail(path, "type must be a string or a nonempty array of strings");
    const auto& t = v.str();
    if (t == "null")
      return kNull;
    if (t == "boolean")
      return kBool;
    if (t == "integer")
      return kInteger;
    if (t == "number")
      return kInteger | kReal;
    if (t == "string")
      return kString;
    if (t == "array")
      return kArray;
    if (t == "object")
      return kObject;
    Fail(path, "unknown type " + t);
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  Id Parse(const Value& v, const std::string& path) {
    CallDepth call(call_depth_, path);
    Step(path);
    ParseDepth guard(parse_depth_);
    if (parse_depth_ > 128)
      Fail(path, "schema reference/parse depth limit exceeded");
    if (v.is_bool())
      return v.as_bool() ? 0 : 1;
    if (!v.is_object())
      Fail(path, "schema must be an object or boolean");
    if (const auto it = parsed_.find(&v); it != parsed_.end())
      return it->second;
    if (program_->schemas.size() >= program_->limits.max_schema_nodes)
      Fail(path, "schema node limit exceeded");
    const Id id = static_cast<Id>(program_->schemas.size());
    parsed_[&v] = id;
    program_->schemas.emplace_back();
    Schema s;
    s.path = path;
    const auto* ref = v.find("$ref");
    const auto* any = v.find("anyOf");
    const auto* one = v.find("oneOf");
    if ((ref != nullptr) + (any != nullptr) + (one != nullptr) > 1)
      Fail(path, "intersections of $ref/anyOf/oneOf are unsupported");
    for (const auto& [key, child] : v.members()) {
      if (Annotation(key))
        continue;
      if (key == "$defs" || key == "definitions") {
        if (!child.is_object())
          Fail(Pointer(path, key), "expected object of schemas");
        for (const auto& [name, definition] : child.members())
          (void)Parse(definition, Pointer(Pointer(path, key), name));
        continue;
      }
      if (key == "$ref" || key == "anyOf" || key == "oneOf")
        continue;
      if (ref || any || one)
        Fail(Pointer(path, key),
             "assertion siblings of $ref/anyOf/oneOf unsupported");
      if (key != "type" && key != "properties" && key != "required" &&
          key != "additionalProperties" && key != "items" &&
          key != "minItems" && key != "maxItems" && key != "minLength" &&
          key != "maxLength" && key != "enum" && key != "const")
        Fail(Pointer(path, key), "unsupported schema keyword");
    }
    if (ref) {
      s.kind = Schema::Kind::kRef;
      s.ref = Parse(Resolve(*ref, path + "/$ref"), ref->str());
    } else if (any || one) {
      s.kind = Schema::Kind::kUnion;
      s.one_of = one != nullptr;
      const auto& choices = one ? *one : *any;
      const auto at = path + (one ? "/oneOf" : "/anyOf");
      if (!choices.is_array() || choices.empty())
        Fail(at, "expected nonempty array of schemas");
      for (std::size_t i = 0; i < choices.size(); ++i)
        s.branches.push_back(
            Parse(choices.items()[i], Pointer(at, std::to_string(i))));
    } else {
      if (const auto* types = v.find("type")) {
        if (types->is_array()) {
          if (types->empty())
            Fail(path + "/type", "empty type array");
          s.types = 0;
          std::set<std::string> seen;
          for (const auto& type : types->items()) {
            const auto mask = ParseType(type, path + "/type");
            if (!seen.insert(type.str()).second)
              Fail(path + "/type", "duplicate type");
            s.types |= mask;
          }
        } else {
          s.types = ParseType(*types, path + "/type");
        }
      }
      if (const auto* extra = v.find("additionalProperties"))
        s.additional = Parse(*extra, path + "/additionalProperties");
      if (const auto* props = v.find("properties")) {
        if (!props->is_object())
          Fail(path + "/properties", "expected object of schemas");
        for (const auto& [name, schema] : props->members())
          s.properties.push_back(
              {name, Parse(schema, Pointer(path + "/properties", name)),
               false});
      }
      if (const auto* required = v.find("required")) {
        if (!required->is_array())
          Fail(path + "/required", "expected array of unique strings");
        std::set<std::string> names;
        for (const auto& name : required->items()) {
          if (!name.is_string() || !names.insert(name.str()).second)
            Fail(path + "/required", "expected array of unique strings");
          auto it = std::find_if(
              s.properties.begin(), s.properties.end(),
              [&](const Property& p) { return p.name == name.str(); });
          if (it == s.properties.end())
            s.properties.push_back({name.str(), s.additional, true});
          else
            it->required = true;
        }
      }
      if (s.properties.size() > program_->limits.max_properties)
        Fail(path + "/properties", "property count limit exceeded");
      if (const auto* items = v.find("items"))
        s.items = Parse(*items, path + "/items");
      if (const auto* value = v.find("minItems"))
        s.min_items =
            Bound(*value, path + "/minItems", program_->limits.max_items);
      if (const auto* value = v.find("maxItems"))
        s.max_items =
            Bound(*value, path + "/maxItems", program_->limits.max_items);
      if (const auto* value = v.find("minLength"))
        s.min_length = Bound(*value, path + "/minLength",
                             program_->limits.max_string_length);
      if (const auto* value = v.find("maxLength"))
        s.max_length = Bound(*value, path + "/maxLength",
                             program_->limits.max_string_length);
      if (const auto* values = v.find("enum")) {
        if (!values->is_array() || values->empty())
          Fail(path + "/enum", "expected nonempty array");
        s.finite = true;
        for (const auto& value : values->items()) {
          if (std::any_of(s.values.begin(), s.values.end(),
                          [&](const Value& existing) {
                            return Equal(existing, value, work_, call_depth_);
                          }))
            Fail(path + "/enum", "duplicate enum value");
          s.values.push_back(value);
        }
      }
      if (const auto* value = v.find("const")) {
        const bool allowed =
            !s.finite ||
            std::any_of(s.values.begin(), s.values.end(),
                        [&](const Value& entry) {
                          return Equal(entry, *value, work_, call_depth_);
                        });
        s.finite = true;
        s.values.clear();
        if (allowed)
          s.values.push_back(*value);
      }
      for (const auto& value : s.values) {
        std::size_t nodes = 0, bytes = 0;
        CheckAst(value, path + "/const|enum", 0, nodes, bytes, true);
      }
    }
    program_->schemas[id] = std::move(s);
    return id;
  }

  // Reference/union cycles with no intervening JSON container are rejected.
  // NOLINTNEXTLINE(misc-no-recursion)
  std::size_t CheckUnguarded(Id id, std::vector<unsigned char>& marks,
                             std::size_t depth = 0) {
    CallDepth call(call_depth_, program_->schemas[id].path);
    Step(program_->schemas[id].path);
    if (depth > 128)
      Fail(program_->schemas[id].path, "schema reference depth limit exceeded");
    if (marks[id] == 2)
      return ref_heights_[id];
    if (marks[id] == 1)
      Fail(program_->schemas[id].path, "unguarded reference cycle");
    marks[id] = 1;
    const auto& s = program_->schemas[id];
    std::size_t height = 0;
    if (s.kind == Schema::Kind::kRef)
      height = 1 + CheckUnguarded(s.ref, marks, depth + 1);
    if (s.kind == Schema::Kind::kUnion)
      for (Id child : s.branches)
        height = std::max(height, 1 + CheckUnguarded(child, marks, depth + 1));
    if (height > 128)
      Fail(s.path, "schema reference depth limit exceeded");
    marks[id] = 2;
    return ref_heights_[id] = height;
  }

  Id Dereference(Id id) const {
    while (program_->schemas[id].kind == Schema::Kind::kRef)
      id = program_->schemas[id].ref;
    return id;
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  unsigned Types(Id id) const {
    CallDepth call(call_depth_, program_->schemas[id].path);
    Step(program_->schemas[id].path);
    id = Dereference(id);
    if (const auto it = type_masks_.find(id); it != type_masks_.end())
      return it->second;
    const auto& s = program_->schemas[id];
    unsigned result = s.types;
    if (s.kind == Schema::Kind::kFalse) {
      result = 0;
    } else if (s.kind == Schema::Kind::kUnion) {
      result = 0;
      for (Id child : s.branches)
        result |= Types(child);
    } else if (s.finite) {
      result = 0;
      for (const auto& v : s.values)
        result |= TypeOf(v);
      result &= s.types;
    }
    type_masks_[id] = result;
    return result;
  }

  // Conservative proof: disjoint types, finite domains, or required tags.
  // NOLINTNEXTLINE(misc-no-recursion)
  bool Disjoint(Id a, Id b, std::size_t depth) const {
    CallDepth call(call_depth_, program_->schemas[a].path);
    Step(program_->schemas[a].path);
    if (depth > program_->limits.max_depth)
      return false;
    a = Dereference(a);
    b = Dereference(b);
    const auto& left = program_->schemas[a];
    const auto& right = program_->schemas[b];
    const auto overlap = Types(a) & Types(b);
    if (overlap == 0)
      return true;
    if (left.kind == Schema::Kind::kUnion)
      return std::all_of(left.branches.begin(), left.branches.end(),
                         [&](Id id) { return Disjoint(id, b, depth + 1); });
    if (right.kind == Schema::Kind::kUnion)
      return std::all_of(right.branches.begin(), right.branches.end(),
                         [&](Id id) { return Disjoint(a, id, depth + 1); });
    if (left.finite) {
      for (const auto& value : left.values)
        if (ProofMatches(a, value) && ProofMatches(b, value))
          return false;
      return true;
    }
    if (right.finite)
      return Disjoint(b, a, depth + 1);
    if (overlap == kObject) {
      for (const auto& p : left.properties) {
        if (!p.required)
          continue;
        for (const auto& q : right.properties)
          if (q.required && p.name == q.name &&
              Disjoint(p.schema, q.schema, depth + 1))
            return true;
      }
    }
    return false;
  }

  Id Make(Expression e) {
    if (program_->expressions.size() >= program_->limits.max_grammar_nodes)
      Fail("#", "compiled grammar node limit exceeded");
    program_->expressions.push_back(std::move(e));
    return static_cast<Id>(program_->expressions.size() - 1);
  }

  Id Literal(std::string value) {
    if (value.empty())
      return 1;
    Expression e;
    e.kind = Expression::Kind::kLiteral;
    e.minimum = value.size();
    e.literal = std::move(value);
    return Make(std::move(e));
  }

  Id Sequence(std::vector<Id> children) {
    Expression e;
    e.kind = Expression::Kind::kSequence;
    for (Id child : children) {
      if (child == 0)
        return 0;
      if (child == 1)
        continue;
      e.children.push_back(child);
      e.minimum = Add(e.minimum, program_->expressions[child].minimum);
    }
    if (e.children.empty())
      return 1;
    if (e.children.size() == 1)
      return e.children[0];
    return Make(std::move(e));
  }

  Id Alternative(std::vector<Id> children) {
    Expression e;
    e.kind = Expression::Kind::kAlternative;
    e.minimum = kInfinity;
    for (Id child : children) {
      if (child == 0 || std::find(e.children.begin(), e.children.end(),
                                  child) != e.children.end())
        continue;
      e.children.push_back(child);
      e.minimum = std::min(e.minimum, program_->expressions[child].minimum);
    }
    if (e.children.empty())
      return 0;
    if (e.children.size() == 1)
      return e.children[0];
    return Make(std::move(e));
  }

  Id Repeat(Id item, Id separator, std::size_t low, std::size_t high) {
    if (low > high || (item == 0 && low != 0))
      return 0;
    if (high == 0 || item == 0)
      return 1;
    Expression e;
    e.kind = Expression::Kind::kRepeat;
    e.children = {item, separator};
    e.low = low;
    e.high = high;
    e.minimum = Add(Multiply(low, program_->expressions[item].minimum),
                    Multiply(low == 0 ? 0 : low - 1,
                             program_->expressions[separator].minimum));
    return Make(std::move(e));
  }

  Id String(std::size_t low, std::size_t high) {
    if (low > high)
      return 0;
    Expression e;
    e.kind = Expression::Kind::kString;
    e.low = low;
    e.high = high;
    e.minimum = Add(low, 2);
    return Make(std::move(e));
  }

  bool LiteralFits(const Value& v, std::size_t depth) const {
    const auto& l = program_->limits;
    std::vector<std::pair<const Value*, std::size_t>> todo{{&v, depth}};
    while (!todo.empty()) {
      const auto [value, d] = todo.back();
      todo.pop_back();
      if ((value->is_object() || value->is_array()) && d >= l.max_depth)
        return false;
      if (value->is_string() &&
          *ScalarLength(value->str()) > l.max_string_length)
        return false;
      if (value->is_number() && value->dump().size() > l.max_number_bytes)
        return false;
      if (value->is_array()) {
        if (value->size() > l.max_items)
          return false;
        for (const auto& child : value->items())
          todo.emplace_back(&child, d + 1);
      }
      if (value->is_object()) {
        if (value->size() > l.max_properties)
          return false;
        for (const auto& [key, child] : value->members()) {
          if (*ScalarLength(key) > l.max_string_length)
            return false;
          todo.emplace_back(&child, d + 1);
        }
      }
    }
    return true;
  }

  // Depth specialization makes guarded recursion finite. Unproductive branches
  // are removed before they can expose a prefix (not at closing-brace time).
  // NOLINTNEXTLINE(misc-no-recursion)
  Id Build(Id id, std::size_t depth) {
    CallDepth call(call_depth_, program_->schemas[id].path);
    Step(program_->schemas[id].path);
    const auto key = std::make_pair(id, depth);
    if (const auto it = built_.find(key); it != built_.end())
      return it->second;
    const auto& s = program_->schemas[id];
    const auto& l = program_->limits;
    Id result = 0;
    if (s.kind == Schema::Kind::kFalse) {
      result = 0;
    } else if (s.kind == Schema::Kind::kRef) {
      result = Build(s.ref, depth);
    } else if (s.kind == Schema::Kind::kUnion) {
      std::vector<Id> children;
      for (Id branch : s.branches)
        children.push_back(Build(branch, depth));
      result = Alternative(std::move(children));
    } else if (s.finite) {
      std::vector<Id> children;
      for (const auto& v : s.values)
        if (ProofMatches(id, v, true) && LiteralFits(v, depth))
          children.push_back(Literal(v.dump()));
      result = Alternative(std::move(children));
    } else {
      std::vector<Id> alternatives;
      if (s.types & kNull)
        alternatives.push_back(Literal("null"));
      if (s.types & kBool) {
        alternatives.push_back(Literal("true"));
        alternatives.push_back(Literal("false"));
      }
      if (s.types & (kInteger | kReal)) {
        Expression number;
        number.kind = Expression::Kind::kNumber;
        number.integer = (s.types & kReal) == 0;
        number.high = l.max_number_bytes;
        number.minimum = 1;
        alternatives.push_back(Make(std::move(number)));
      }
      if (s.types & kString)
        alternatives.push_back(
            String(s.min_length, std::min(s.max_length, l.max_string_length)));
      if ((s.types & kArray) && depth < l.max_depth) {
        const Id item = Sequence({Build(s.items, depth + 1), whitespace_});
        const Id repeat =
            Repeat(item, Sequence({Literal(","), whitespace_}), s.min_items,
                   std::min(s.max_items, l.max_items));
        alternatives.push_back(
            Sequence({Literal("["), whitespace_, repeat, Literal("]")}));
      }
      if ((s.types & kObject) && depth < l.max_depth) {
        Id body;
        if (s.properties.empty()) {
          const Id item = Sequence(
              {String(0, l.max_string_length), whitespace_, Literal(":"),
               whitespace_, Build(s.additional, depth + 1), whitespace_});
          body = Repeat(item, Sequence({Literal(","), whitespace_}), 0,
                        l.max_properties);
        } else {
          Id empty_tail = 1, comma_tail = 1;
          for (auto it = s.properties.rbegin(); it != s.properties.rend();
               ++it) {
            if (*ScalarLength(it->name) > l.max_string_length)
              Fail(Pointer(s.path + "/properties", it->name),
                   "property name exceeds string limit");
            const Id member = Sequence(
                {Literal(Value(it->name).dump()), whitespace_, Literal(":"),
                 whitespace_, Build(it->schema, depth + 1), whitespace_});
            const Id without_comma = Sequence({member, comma_tail});
            const Id with_comma =
                Sequence({Literal(","), whitespace_, member, comma_tail});
            empty_tail = it->required
                             ? without_comma
                             : Alternative({without_comma, empty_tail});
            comma_tail = it->required ? with_comma
                                      : Alternative({with_comma, comma_tail});
          }
          body = empty_tail;
        }
        alternatives.push_back(
            Sequence({Literal("{"), whitespace_, body, Literal("}")}));
      }
      result = Alternative(std::move(alternatives));
    }
    built_[key] = result;
    return result;
  }
};

// A stack frame is either a grammar expansion or a small streaming lexical DFA.
// Counts refer to Unicode scalars, number bytes, or completed repeated items.
struct Frame {
  Id node = 0;
  std::uint32_t count = 0, code = 0;
  std::uint8_t mode = 0, digits = 0, low = 0, high = 0;
  bool operator==(const Frame&) const = default;
};
using Stack = std::vector<Frame>;
struct StackHash {
  std::size_t operator()(const Stack& stack) const noexcept {
    std::size_t hash = 0;
    for (const auto& f : stack) {
      for (auto n : {f.node, f.count, f.code,
                     static_cast<unsigned>(f.mode) |
                         (static_cast<unsigned>(f.digits) << 8) |
                         (static_cast<unsigned>(f.low) << 16) |
                         (static_cast<unsigned>(f.high) << 24)})
        hash ^= n + 0x9e3779b9U + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

enum StringMode : unsigned char {
  kOpen,
  kBody,
  kEscape,
  kHex,
  kHighSlash,
  kLowU,
  kLowHex,
  kUtf8
};
enum NumberMode : unsigned char {
  kStart,
  kMinus,
  kZero,
  kDigits,
  kDot,
  kFraction,
  kExponent,
  kSign,
  kExponentDigits
};

bool NumberComplete(unsigned mode) {
  return mode == kZero || mode == kDigits || mode == kFraction ||
         mode == kExponentDigits;
}

bool HexInterval(unsigned code, unsigned digits, bool low) {
  const unsigned shift = 4 * (4 - digits);
  const unsigned first = code << shift;
  const unsigned last = first + ((1U << shift) - 1);
  return low ? first <= 0xDFFF && last >= 0xDC00
             : first < 0xDC00 || last > 0xDFFF;
}

bool ForcedHigh(unsigned code, unsigned digits) {
  const unsigned shift = 4 * (4 - digits);
  const unsigned first = code << shift;
  const unsigned last = first + ((1U << shift) - 1);
  return first >= 0xD800 && last <= 0xDBFF;
}

std::size_t Minimum(const Program& p, const Frame& f) {
  const auto& e = p.expressions[f.node];
  using K = Expression::Kind;
  if (e.kind == K::kLiteral)
    return e.literal.size() - f.count;
  if (e.kind == K::kWhitespace)
    return 0;
  if (e.kind == K::kNumber)
    return NumberComplete(f.mode) ? 0 : 1;
  if (e.kind == K::kRepeat) {
    const std::size_t count = e.low > f.count ? e.low - f.count : 0;
    const std::size_t separators =
        count == 0 ? 0 : count - (f.count == 0 ? 1 : 0);
    return Add(Multiply(count, p.expressions[e.children[0]].minimum),
               Multiply(separators, p.expressions[e.children[1]].minimum));
  }
  if (e.kind != K::kString)
    return e.minimum;
  if (f.mode == kOpen)
    return e.minimum;
  const std::size_t pending = f.mode == kBody ? 0 : 1;
  const std::size_t scalars = f.count + pending;
  const std::size_t extra = e.low > scalars ? e.low - scalars : 0;
  std::size_t bytes = 0;
  switch (f.mode) {
    case kBody:
      break;
    case kEscape:
      bytes = 1;
      break;
    case kHex:
      bytes = 4 - f.digits + (ForcedHigh(f.code, f.digits) ? 6 : 0);
      break;
    case kHighSlash:
      bytes = 6;
      break;
    case kLowU:
      bytes = 5;
      break;
    case kLowHex:
      bytes = 4 - f.digits;
      break;
    case kUtf8:
      bytes = f.digits;
      break;
    default:
      return kInfinity;
  }
  return Add(Add(bytes, extra), 1);
}

std::size_t Minimum(const Program& p, const Stack& stack) {
  std::size_t result = 0;
  for (const auto& f : stack)
    result = Add(result, Minimum(p, f));
  return result;
}

// A nullable lexical frame is not a consuming state if its only viable path is
// epsilon. This distinction is essential when the frontier/budget is capped.
std::size_t ConsumingMinimum(const Expression& e, const Frame& f) {
  if (f.count >= e.high)
    return kInfinity;
  if (e.kind == Expression::Kind::kWhitespace)
    return 1;
  if (f.mode == kZero) {
    if (e.integer || e.high - f.count < 2)
      return kInfinity;
    return 2;
  }
  return 1;
}

// Epsilon closure only visits productive stacks. On an ambiguity limit, keep a
// deterministic viable subset rather than accepting a prefix that later fails
// from resource exhaustion. Every retained stack has a known finite completion.
std::vector<Stack> Normalize(const Program& p, std::vector<Stack> pending,
                             std::size_t remaining) {
  std::vector<Stack> ready;
  std::unordered_set<Stack, StackHash> seen;
  const std::size_t budget = p.limits.max_states;
  while (!pending.empty() && ready.size() < budget) {
    Stack stack = std::move(pending.back());
    pending.pop_back();
    if (Minimum(p, stack) > remaining || !seen.insert(stack).second)
      continue;
    if (stack.empty()) {
      ready.push_back(std::move(stack));
      continue;
    }
    const Frame frame = stack.back();
    const auto& e = p.expressions[frame.node];
    using K = Expression::Kind;
    if (e.kind == K::kEmpty) {
      stack.pop_back();
      pending.push_back(std::move(stack));
    } else if (e.kind == K::kSequence) {
      stack.pop_back();
      for (auto it = e.children.rbegin(); it != e.children.rend(); ++it)
        stack.push_back(Frame{*it});
      pending.push_back(std::move(stack));
    } else if (e.kind == K::kAlternative) {
      stack.pop_back();
      for (auto it = e.children.rbegin(); it != e.children.rend(); ++it) {
        Stack branch = stack;
        branch.push_back(Frame{*it});
        pending.push_back(std::move(branch));
      }
    } else if (e.kind == K::kRepeat) {
      stack.pop_back();
      if (frame.count < e.high) {
        Stack branch = stack;
        Frame next = frame;
        ++next.count;
        branch.push_back(next);
        branch.push_back(Frame{e.children[0]});
        if (frame.count != 0)
          branch.push_back(Frame{e.children[1]});
        pending.push_back(std::move(branch));
      }
      if (frame.count >= e.low)
        pending.push_back(std::move(stack));
    } else if (e.kind == K::kWhitespace ||
               (e.kind == K::kNumber && NumberComplete(frame.mode))) {
      stack.pop_back();
      if (Add(Minimum(p, stack), ConsumingMinimum(e, frame)) <= remaining) {
        Stack consuming = stack;
        consuming.push_back(frame);
        ready.push_back(std::move(consuming));
      }
      pending.push_back(std::move(stack));
    } else if (e.kind != K::kFail) {
      ready.push_back(std::move(stack));
    }
  }
  return ready;
}

bool StringByte(const Expression& e, Frame& f, unsigned char c, bool& done) {
  done = false;
  if (f.mode == kOpen) {
    if (c != '"')
      return false;
    f.mode = kBody;
  } else if (f.mode == kBody) {
    if (c == '"') {
      done = f.count >= e.low;
      return done;
    }
    if (f.count >= e.high || c < 0x20)
      return false;
    if (c == '\\') {
      f.mode = kEscape;
    } else if (c < 0x80) {
      ++f.count;
    } else {
      f.digits = c >= 0xC2 && c <= 0xDF   ? 1
                 : c >= 0xE0 && c <= 0xEF ? 2
                 : c >= 0xF0 && c <= 0xF4 ? 3
                                          : 0;
      if (!f.digits)
        return false;
      f.mode = kUtf8;
      f.low = c == 0xE0 ? 0xA0 : c == 0xF0 ? 0x90 : 0x80;
      f.high = c == 0xED ? 0x9F : c == 0xF4 ? 0x8F : 0xBF;
    }
  } else if (f.mode == kUtf8) {
    if (c < f.low || c > f.high)
      return false;
    f.low = 0x80;
    f.high = 0xBF;
    if (--f.digits == 0) {
      f.mode = kBody;
      ++f.count;
    }
  } else if (f.mode == kEscape) {
    if (c == 'u') {
      f.mode = kHex;
      f.code = 0;
      f.digits = 0;
    } else if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' ||
               c == 'n' || c == 'r' || c == 't') {
      f.mode = kBody;
      ++f.count;
    } else {
      return false;
    }
  } else if (f.mode == kHighSlash) {
    if (c != '\\')
      return false;
    f.mode = kLowU;
  } else if (f.mode == kLowU) {
    if (c != 'u')
      return false;
    f.mode = kLowHex;
    f.code = 0;
    f.digits = 0;
  } else {
    const unsigned digit = c >= '0' && c <= '9'   ? c - '0'
                           : c >= 'a' && c <= 'f' ? c - 'a' + 10
                           : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                                  : 16;
    if (digit == 16)
      return false;
    f.code = (f.code << 4) | digit;
    ++f.digits;
    if (!HexInterval(f.code, f.digits, f.mode == kLowHex))
      return false;
    if (f.digits == 4) {
      if (f.mode == kHex && f.code >= 0xD800 && f.code <= 0xDBFF) {
        f.mode = kHighSlash;
      } else {
        f.mode = kBody;
        ++f.count;
      }
    }
  }
  return true;
}

bool NumberByte(const Expression& e, Frame& f, unsigned char c) {
  const bool digit = c >= '0' && c <= '9';
  unsigned char next = f.mode;
  switch (f.mode) {
    case kStart:
    case kMinus:
      if (f.mode == kStart && c == '-')
        next = kMinus;
      else if (c == '0')
        next = kZero;
      else if (c >= '1' && c <= '9')
        next = kDigits;
      else
        return false;
      break;
    case kZero:
    case kDigits:
      if (f.mode == kDigits && digit)
        next = kDigits;
      else if (!e.integer && c == '.')
        next = kDot;
      else if (!e.integer && (c == 'e' || c == 'E'))
        next = kExponent;
      else
        return false;
      break;
    case kDot:
    case kFraction:
      if (digit)
        next = kFraction;
      else if (f.mode == kFraction && (c == 'e' || c == 'E'))
        next = kExponent;
      else
        return false;
      break;
    case kExponent:
    case kSign:
    case kExponentDigits:
      if (digit)
        next = kExponentDigits;
      else if (f.mode == kExponent && (c == '+' || c == '-'))
        next = kSign;
      else
        return false;
      break;
    default:
      return false;
  }
  ++f.count;
  f.mode = next;
  return f.count + (NumberComplete(next) ? 0 : 1) <= e.high;
}

struct Cursor {
  std::vector<Stack> states;
  std::size_t bytes = 0;

  bool Advance(const Program& p, std::string_view input) {
    if (input.empty() || input.size() > p.limits.max_output_bytes - bytes)
      return false;
    for (unsigned char c : input) {
      std::vector<Stack> next;
      for (const auto& state : states) {
        if (state.empty())
          continue;
        Stack stack = state;
        auto& f = stack.back();
        const auto& e = p.expressions[f.node];
        bool allowed = false, done = false;
        using K = Expression::Kind;
        if (e.kind == K::kLiteral) {
          allowed = static_cast<unsigned char>(e.literal[f.count]) == c;
          done = allowed && ++f.count == e.literal.size();
        } else if (e.kind == K::kString) {
          allowed = StringByte(e, f, c, done);
        } else if (e.kind == K::kNumber) {
          allowed = NumberByte(e, f, c);
        } else if (e.kind == K::kWhitespace) {
          allowed = (c == ' ' || c == '\t' || c == '\n' || c == '\r') &&
                    f.count < e.high;
          ++f.count;
        }
        if (allowed) {
          if (done)
            stack.pop_back();
          next.push_back(std::move(stack));
        }
      }
      ++bytes;
      states = Normalize(p, std::move(next), p.limits.max_output_bytes - bytes);
      if (states.empty())
        return false;
    }
    return true;
  }
};

}  // namespace

struct JsonConstraint::Impl {
  std::shared_ptr<const Program> program;
  Cursor cursor;
};

JsonConstraint::JsonConstraint(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
JsonConstraint::JsonConstraint(const JsonConstraint& other)
    : impl_(other.impl_ ? std::make_unique<Impl>(*other.impl_) : nullptr) {}
JsonConstraint& JsonConstraint::operator=(const JsonConstraint& other) {
  if (this != &other) {
    auto next = other.impl_ ? std::make_unique<Impl>(*other.impl_) : nullptr;
    impl_ = std::move(next);
  }
  return *this;
}
JsonConstraint::JsonConstraint(JsonConstraint&&) noexcept = default;
JsonConstraint& JsonConstraint::operator=(JsonConstraint&&) noexcept = default;
JsonConstraint::~JsonConstraint() = default;

JsonConstraint JsonConstraint::Compile(const json::Value& schema) {
  return Compile(schema, Limits{});
}
JsonConstraint JsonConstraint::Compile(const json::Value& schema,
                                       const Limits& limits) {
  auto impl = std::make_unique<Impl>();
  try {
    impl->program = Compiler(schema, limits).Compile();
  } catch (const ValidationLimit&) {
    Fail("#", "schema proof/validation work or combined depth limit exceeded");
  }
  impl->cursor.states = Normalize(
      *impl->program, {{Frame{impl->program->root}}}, limits.max_output_bytes);
  if (impl->cursor.states.empty())
    Fail("#", "schema has no completion within safety limits");
  return JsonConstraint(std::move(impl));
}
JsonConstraint JsonConstraint::Compile(std::string_view schema_json) {
  return Compile(schema_json, Limits{});
}
JsonConstraint JsonConstraint::Compile(const char* schema_json) {
  if (!schema_json)
    Fail("#", "null schema text");
  return Compile(std::string_view(schema_json));
}
JsonConstraint JsonConstraint::Compile(std::string_view schema_json,
                                       const Limits& limits) {
  if (schema_json.size() > limits.max_schema_bytes)
    Fail("#", "schema byte limit exceeded");
  Value value;
  try {
    value = json::parse(schema_json);
  } catch (const std::runtime_error& error) {
    Fail("#", error.what());
  }
  return Compile(value, limits);
}
JsonConstraint JsonConstraint::JsonObject() {
  return JsonObject(Limits{});
}
JsonConstraint JsonConstraint::JsonObject(const Limits& limits) {
  Value schema = Value::object();
  schema["type"] = "object";
  return Compile(schema, limits);
}
JsonConstraint JsonConstraint::Clone() const {
  return *this;
}
bool JsonConstraint::Allows(std::string_view bytes) const {
  if (!impl_)
    return false;
  Cursor next = impl_->cursor;
  return next.Advance(*impl_->program, bytes);
}
void JsonConstraint::Accept(std::string_view bytes) {
  if (!impl_)
    throw std::invalid_argument("JSON constraint: moved-from cursor");
  Cursor next = impl_->cursor;
  if (!next.Advance(*impl_->program, bytes))
    throw std::invalid_argument(
        "JSON constraint: token has no viable completion");
  impl_->cursor = std::move(next);
}
bool JsonConstraint::Complete() const {
  return impl_ &&
         std::any_of(impl_->cursor.states.begin(), impl_->cursor.states.end(),
                     [](const Stack& stack) { return stack.empty(); });
}
bool JsonConstraint::Validate(const json::Value& value) const {
  if (!impl_)
    return false;
  try {
    return impl_->program->Validate(impl_->program->schema_root, value);
  } catch (const ValidationLimit&) {
    return false;
  }
}

}  // namespace gufo
