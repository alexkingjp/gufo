#include "src/core/json_constraint.hpp"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using gufo::JsonConstraint;
using gufo::json::parse;

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "JSON constraint check failed: " << message << '\n';
    std::abort();
  }
}

void RejectSchema(std::string_view schema, std::string_view error_part) {
  try {
    (void)JsonConstraint::Compile(schema);
  } catch (const std::invalid_argument& error) {
    Check(std::string_view(error.what()).find(error_part) !=
              std::string_view::npos,
          error.what());
    return;
  }
  Check(false, schema);
}

void AcceptAll(const JsonConstraint& prototype, std::string_view text) {
  auto whole = prototype.Clone();
  Check(whole.Allows(text), text);
  whole.Accept(text);
  Check(whole.Complete(), text);
  Check(!whole.Allows("x"), "trailing garbage rejected");
  for (std::size_t split = 0; split <= text.size(); ++split) {
    auto cursor = prototype;
    if (split != 0) {
      Check(cursor.Allows(text.substr(0, split)), text);
      cursor.Accept(text.substr(0, split));
    }
    if (split != text.size()) {
      Check(cursor.Allows(text.substr(split)), text);
      cursor.Accept(text.substr(split));
    }
    Check(cursor.Complete(), "all token boundaries complete");
  }
  auto bytes = prototype;
  for (std::size_t i = 0; i < text.size(); ++i) {
    const auto byte = text.substr(i, 1);
    Check(bytes.Allows(byte), text);
    bytes.Accept(byte);
  }
  Check(bytes.Complete(), "bytewise completion");
}

void TestGeneric() {
  const auto object = JsonConstraint::JsonObject();
  Check(!object.Complete(), "empty cursor is not complete");
  Check(!object.Allows(""), "empty tokens are disallowed");
  Check(!object.Allows("null"), "json_object excludes scalar root");
  for (const auto text :
       {"{}", " { \n}\t", R"({"x":[true,false,null,-1.25e+2,{"y":"z"}]})",
        R"({"arbitrary":1,"other":2,"last":[]})"})
    AcceptAll(object, text);
  for (const auto text : {"{]", "{,", R"({"x":01})", R"({"x":1,})",
                          R"({"x":true false})", R"({"x" 1})", "{}{}"})
    Check(!object.Allows(text), text);
  for (const auto schema : {"true", "{}"}) {
    const auto any = JsonConstraint::Compile(schema);
    for (const auto text :
         {"null", "false", "0", "-3.5e-2", "\"\"", "[]", "{}"})
      AcceptAll(any, text);
  }
  RejectSchema("false", "no finite completion");
}

void TestNumbers() {
  const auto number = JsonConstraint::Compile(R"({"type":"number"})");
  for (const auto text :
       {"0", "-0", "12", "-123", "0.1", "1e2", "1E-2", "1e+2", "1.25E+10"})
    AcceptAll(number, text);
  for (const auto prefix : {"-", "1.", "1e", "1e+", "1e-"}) {
    auto cursor = number;
    Check(cursor.Allows(prefix), prefix);
    cursor.Accept(prefix);
    Check(!cursor.Complete(), "incomplete numeric lexeme is not EOS");
    cursor.Accept("1");
    Check(cursor.Complete(), "numeric suffix completes lexeme");
  }
  for (const auto text :
       {"+1", "01", "-01", ".1", "1.e2", "1e+e", "NaN", "Infinity", "1 2"})
    Check(!number.Allows(text), text);
  const auto integer = JsonConstraint::Compile(R"({"type":"integer"})");
  AcceptAll(integer, "-123");
  Check(!integer.Allows("1."), "integer uses integer spelling");
  Check(integer.Validate(parse("1.0")),
        "integer structural validation is mathematical");
  Check(!integer.Validate(parse("1.5")), "noninteger rejected");
  Check(!integer.Validate(parse("1.0000000000000001")),
        "rounded fraction cannot satisfy integer");
  auto zero = number;
  zero.Accept("0");
  Check(zero.Complete() && !zero.Allows("1") && zero.Allows(".1"),
        "leading-zero rule");
}

void TestStrings() {
  const auto str = JsonConstraint::Compile(
      R"({"type":"string","minLength":1,"maxLength":2})");
  for (const auto text : {"\"a\"", "\"ab\"", "\"\xC2\xA2\"", "\"\xE2\x82\xAC\"",
                          "\"\xF0\x9F\x98\x80\"", R"("\uD83D\uDE00")",
                          R"("\n\t")", R"("\u0061\/")"})
    AcceptAll(str, text);
  for (const auto text : {R"("")", R"("abc")", R"("\q")", R"("\uDC)",
                          R"("\uD800x)", R"("\uD800\u0)", "\"\xC0", "\"\x80",
                          "\"\xED\xA0", "\"\xF4\x90", "\"\n"})
    Check(!str.Allows(text), text);
  auto surrogate = str;
  surrogate.Accept(R"("\uD83D)");
  Check(!surrogate.Complete(), "unpaired high surrogate incomplete");
  Check(!surrogate.Allows("\""), "surrogate cannot close string");
  surrogate.Accept(R"(\uDE00")");
  Check(surrogate.Complete(), "surrogate pair completes");
  const auto empty =
      JsonConstraint::Compile(R"({"type":"string","maxLength":0})");
  AcceptAll(empty, R"("")");
  Check(!empty.Allows(R"("\)"), "escape would exceed maxLength");
  Check(str.Validate(parse(R"("\uD83D\uDE00")")), "scalar length validation");
  Check(!str.Validate(parse(R"("abc")")), "length validation");
}

void TestObjectsAndArrays() {
  const auto schema = JsonConstraint::Compile(R"({"type":"object","properties":{
    "optional":{"type":"string"},"required":{"type":"integer"},
    "later":{"type":"array","items":{"type":"boolean"},"minItems":1,"maxItems":2}},
    "required":["required"],"additionalProperties":false})");
  for (const auto text :
       {R"({"required":1})", R"({"optional":"ok","required":0})",
        R"({"required":2,"later":[false]})",
        R"({"optional":"","required":3,"later":[true,false]})"})
    AcceptAll(schema, text);
  for (const auto text :
       {"{}", R"({"optional":"x"})", R"({"required":1,"unknown":0})",
        R"({"required":1,"later":[]})",
        R"({"required":1,"later":[true,false,true]})",
        R"({"later":[true],"required":1})"})
    Check(!schema.Allows(text), text);
  Check(
      schema.Validate(parse(R"({"later":[true],"required":1,"optional":"x"})")),
      "validation ignores property order");
  Check(!schema.Validate(parse(R"({"required":1,"extra":2})")),
        "additionalProperties enforced in validation");
  Check(!schema.Validate(parse("{}")), "required enforced in validation");
  const auto false_optional = JsonConstraint::Compile(
      R"({"type":"object","properties":{"bad":false,"ok":{"type":"null"}},"required":["ok"],"additionalProperties":false})");
  AcceptAll(false_optional, R"({"ok":null})");
  Check(!false_optional.Allows(R"({"b)"),
        "unproductive optional property not exposed");
  const auto missing = JsonConstraint::Compile(
      R"({"type":"object","required":["x"],"additionalProperties":{"type":"boolean"}})");
  AcceptAll(missing, R"({"x":true})");
  Check(missing.Validate(parse(R"({"y":false,"x":true})")),
        "schema-valued additionalProperties");
  RejectSchema(
      R"({"type":"object","required":["x"],"additionalProperties":false})",
      "no finite completion");
  RejectSchema(R"({"type":"object","properties":{"x":false},"required":["x"]})",
               "no finite completion");
  const auto array = JsonConstraint::Compile(
      R"({"type":"array","items":{"type":"array","items":{"type":"null"},"minItems":1,"maxItems":1},"minItems":1,"maxItems":2})");
  AcceptAll(array, "[[null],[null]]");
  Check(!array.Allows("[[]"), "nested minItems");
  Check(!array.Allows("[[null],[null],"), "maxItems rejects comma early");
  const auto no_items =
      JsonConstraint::Compile(R"({"type":"array","items":false})");
  AcceptAll(no_items, "[]");
  Check(!no_items.Allows("[n"), "false items cannot start");
  RejectSchema(R"({"type":"array","items":false,"minItems":1})",
               "no finite completion");
  RejectSchema(R"({"type":"string","minLength":3,"maxLength":2})",
               "no finite completion");
}

void TestAlternatives() {
  const auto schema = JsonConstraint::Compile(
      R"({"anyOf":[false,{"type":"string","enum":["a","ab"]},{"type":"null"},{"type":"array","items":{"type":"integer"},"maxItems":2}]})");
  for (const auto text : {R"("a")", R"("ab")", "null", "[1,2]"})
    AcceptAll(schema, text);
  Check(!schema.Allows(R"("ac)"), "enum prefix rejected");
  const auto finite =
      JsonConstraint::Compile(R"({"enum":[{"b":2,"a":1},[false],7]})");
  AcceptAll(finite, R"({"b":2,"a":1})");
  Check(finite.Validate(parse(R"({"a":1,"b":2})")),
        "enum structural object equality");
  const auto one = JsonConstraint::Compile(
      R"({"oneOf":[{"type":"integer"},{"type":"string"},{"type":"null"}]})");
  AcceptAll(one, "1");
  AcceptAll(one, "null");
  const auto tagged = JsonConstraint::Compile(
      R"({"oneOf":[{"type":"object","properties":{"tag":{"const":"a"}},"required":["tag"]},{"type":"object","properties":{"tag":{"const":"b"}},"required":["tag"]}]})");
  AcceptAll(tagged, R"({"tag":"b"})");
  RejectSchema(R"({"oneOf":[{"type":"integer"},{"type":"number"}]})",
               "cannot be proven disjoint");
  RejectSchema(R"({"oneOf":[{"const":1},{"const":1}]})", "overlap");
  const auto types =
      JsonConstraint::Compile(R"({"type":["string","null"],"minLength":2})");
  AcceptAll(types, "null");
  AcceptAll(types, R"("ab")");
  Check(!types.Allows(R"("a")"), "type union retains assertions");
  const auto intersect = JsonConstraint::Compile(
      R"({"type":"integer","enum":["x",2,3],"const":2})");
  AcceptAll(intersect, "2");
  Check(!intersect.Allows("3"), "enum/const/type intersection");
}

void TestReferences() {
  const auto schema = JsonConstraint::Compile(
      R"({"$defs":{"entry":{"type":"object","properties":{"value":{"type":"integer"},"next":{"anyOf":[{"type":"null"},{"$ref":"#/$defs/entry"}]}},"required":["value","next"],"additionalProperties":false}},"$ref":"#/$defs/entry"})");
  AcceptAll(schema, R"({"value":1,"next":{"value":2,"next":null}})");
  Check(schema.Validate(parse(R"({"next":null,"value":0})")),
        "recursive validator");
  const auto pointer = JsonConstraint::Compile(
      R"({"$defs":{"a/b~c":{"type":"boolean"}},"$ref":"#/$defs/a~1b~0c"})");
  AcceptAll(pointer, "false");
  RejectSchema(R"({"$ref":"#"})", "unguarded reference cycle");
  RejectSchema(
      R"({"$defs":{"a":{"$ref":"#/$defs/b"},"b":{"$ref":"#/$defs/a"}},"$ref":"#/$defs/a"})",
      "unguarded reference cycle");
  RejectSchema(
      R"({"type":"object","properties":{"next":{"$ref":"#"}},"required":["next"]})",
      "no finite completion");
  RejectSchema(R"({"$ref":"#/$defs/missing"})", "unresolved");
  RejectSchema(R"({"$ref":"https://example.com/schema"})", "only local");
}

void TestLimitsAndState() {
  auto limits = JsonConstraint::Limits{};
  limits.max_states = 1;
  limits.max_output_bytes = 5;
  auto budgeted = JsonConstraint::Compile(parse(R"({"type":"null"})"), limits);
  Check(budgeted.Allows(" "), "bounded frontier allows viable whitespace");
  budgeted.Accept(" ");
  Check(budgeted.Allows("null"),
        "bounded frontier retains completion after whitespace");
  budgeted.Accept("null");
  Check(budgeted.Complete(), "bounded frontier keeps EOS at byte budget");
  limits.max_output_bytes = 100;
  auto zero_only =
      JsonConstraint::Compile(parse(R"({"type":"integer"})"), limits);
  // A saturated frontier may choose bounded whitespace before the number.
  zero_only.Accept(std::string(limits.max_whitespace, ' ') + "0");
  zero_only.Accept(std::string(limits.max_whitespace, ' '));
  Check(zero_only.Complete(), "integer zero has no spurious consuming state");
  limits = JsonConstraint::Limits{};
  limits.max_output_bytes = 6;
  limits.max_whitespace = 1;
  const auto str = JsonConstraint::Compile(
      parse(R"({"type":"string","minLength":2})"), limits);
  Check(!str.Allows(" \"\xF0"), "UTF-8 prefix reserves completion budget");
  AcceptAll(str, " \"ab\" ");
  auto capped = str;
  capped.Accept("\"abcd");
  Check(!capped.Allows("e"), "output limit reserves closing quote");
  capped.Accept("\"");
  Check(capped.Complete(), "completion at exact byte budget");
  limits.max_output_bytes = 100;
  limits.max_number_bytes = 2;
  auto number = JsonConstraint::Compile(parse(R"({"type":"number"})"), limits);
  Check(!number.Allows("1e"), "number byte limit reserves exponent digit");
  AcceptAll(number, "-1");
  limits.max_depth = 2;
  auto recursive = JsonConstraint::Compile(
      parse(R"({"type":"array","items":{"$ref":"#"}})"), limits);
  AcceptAll(recursive, "[[]]");
  Check(!recursive.Allows("[[["),
        "depth bound rejects opening impossible child");
  limits.max_whitespace = 0;
  auto obj = JsonConstraint::JsonObject(limits);
  AcceptAll(obj, "{}");
  Check(!obj.Allows(" "), "zero whitespace limit");
  auto cursor = JsonConstraint::Compile(R"({"type":"string"})");
  cursor.Accept("\"a");
  auto copy = cursor;
  Check(!cursor.Allows("\n"), "Allows rejects without mutation");
  bool threw = false;
  try {
    cursor.Accept("b\n");
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Check(threw, "Accept rejects invalid token");
  cursor.Accept("c\"");
  copy.Accept("d\"");
  Check(cursor.Complete() && copy.Complete(), "copies independently advance");
}

void TestProofAndResourceRegressions() {
  auto limits = JsonConstraint::Limits{};
  limits.max_output_bytes = 3;
  bool rejected = false;
  try {
    (void)JsonConstraint::Compile(
        parse(
            R"({"oneOf":[{"const":[1]},{"type":"array","items":{"$ref":"#/$defs/a"}}],"$defs":{"a":{"$ref":"#/$defs/b"},"b":{"type":"integer"}}})"),
        limits);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Check(rejected, "tiny output budget cannot prove overlapping oneOf disjoint");

  auto dag = parse(
      R"({"$defs":{"n0":{"type":"null"}},"oneOf":[{"$ref":"#/$defs/n30"},{"type":"string"}]})");
  for (int i = 1; i <= 30; ++i) {
    auto ref = gufo::json::Value::object();
    ref["$ref"] = "#/$defs/n" + std::to_string(i - 1);
    auto alternatives = gufo::json::Value::array();
    alternatives.push_back(ref);
    alternatives.push_back(ref);
    auto node = gufo::json::Value::object();
    node["anyOf"] = alternatives;
    dag["$defs"]["n" + std::to_string(i)] = node;
  }
  auto compiled_dag = JsonConstraint::Compile(dag);
  AcceptAll(compiled_dag, "null");
  AcceptAll(compiled_dag, R"("ok")");

  dag["$defs"]["n0"]["type"] = "string";
  dag["oneOf"] = parse(
      R"([{ "const":[1] },{"type":"array","items":{"anyOf":[{"$ref":"#/$defs/n30"},{"type":"integer"}]}}])");
  rejected = false;
  try {
    (void)JsonConstraint::Compile(dag);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Check(rejected, "validation exhaustion cannot prove disjoint oneOf");

  auto chain = parse(R"({"$defs":{},"$ref":"#/$defs/n0"})");
  for (int i = 0; i < 3000; ++i) {
    auto ref = gufo::json::Value::object();
    ref["$ref"] = "#/$defs/n" + std::to_string(i + 1);
    chain["$defs"]["n" + std::to_string(i)] = ref;
  }
  chain["$defs"]["n3000"] = parse(R"({"type":"null"})");
  rejected = false;
  try {
    (void)JsonConstraint::Compile(chain);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Check(rejected, "long reference chain rejected without stack exhaustion");

  RejectSchema(R"({"const":1.0000000000000001})", "loses precision");
  RejectSchema(R"({"enum":[0.1,0.10000000000000001]})", "loses precision");
  RejectSchema(parse(R"({"const":1.0000000000000001})").dump(),
               "loses precision");
  auto exact = JsonConstraint::Compile(R"({"const":0.1})");
  AcceptAll(exact, "0.1");
  Check(!exact.Validate(parse("0.10000000000000001")),
        "inexact input does not match numeric const");
}

void TestSharedProofBudgetAndCombinedDepth() {
  auto finite = parse(R"({"oneOf":[]})");
  for (int i = 0; i < 40; ++i) {
    auto branch = gufo::json::Value::object();
    branch["const"] = i;
    finite["oneOf"].push_back(branch);
  }
  auto generous = JsonConstraint::Compile(finite);
  AcceptAll(generous, "17");
  auto limits = JsonConstraint::Limits{};
  limits.max_compile_steps = 4000;
  bool rejected = false;
  try {
    (void)JsonConstraint::Compile(finite, limits);
  } catch (const std::invalid_argument& error) {
    rejected =
        std::string_view(error.what()).find("work") != std::string_view::npos;
  }
  Check(rejected, "finite branch proofs share one global compilation budget");
  (void)JsonConstraint::Compile(parse(R"({"const":17})"), limits);

  auto recursive = parse(
      R"({"$defs":{"n0":{"type":"array","items":{"$ref":"#/$defs/n10"}}},"$ref":"#/$defs/n10"})");
  for (int i = 1; i <= 10; ++i) {
    auto ref = gufo::json::Value::object();
    ref["$ref"] = "#/$defs/n" + std::to_string(i - 1);
    recursive["$defs"]["n" + std::to_string(i)] = ref;
  }
  limits = JsonConstraint::Limits{};
  limits.max_depth = 10;
  auto shallow = JsonConstraint::Compile(recursive, limits);
  AcceptAll(shallow, "[[]]");
  limits.max_depth = 32;
  rejected = false;
  try {
    (void)JsonConstraint::Compile(recursive, limits);
  } catch (const std::invalid_argument& error) {
    rejected = std::string_view(error.what()).find("combined") !=
               std::string_view::npos;
  }
  Check(rejected, "container and reference recursion share the 256-frame cap");
}

void TestSchemaErrors() {
  for (const auto key :
       {"pattern", "format", "minimum", "maximum", "multipleOf", "allOf", "not",
        "uniqueItems", "patternProperties", "unevaluatedProperties", "$id"}) {
    RejectSchema(std::string("{\"") + key + "\":true}",
                 std::string("#/") + key);
  }
  for (const auto schema :
       {R"({"type":"wat"})", R"({"type":[]})", R"({"type":["null","null"]})",
        R"({"required":["x","x"]})", R"({"properties":[]})", R"({"items":[]})",
        R"({"minItems":-1})", R"({"maxLength":1.5})",
        R"({"maxItems":1.0000000000000001})", R"({"enum":[]})",
        R"({"enum":[1,1.0]})", R"({"const":9007199254740992})", "[]", "{",
        "null"})
    RejectSchema(schema, "JSON constraint #");
  RejectSchema(R"({"anyOf":[true],"type":"string"})", "assertion siblings");
  RejectSchema(R"({"type":"object","$defs":{"unused":{"pattern":"x"}}})",
               "pattern");
  const auto annotated = JsonConstraint::Compile(
      R"({"type":"boolean","title":"Choice","description":"A flag","default":false,"examples":[true],"$comment":"ok","readOnly":true})");
  AcceptAll(annotated, "true");
}
}  // namespace

int main() {
  TestGeneric();
  TestNumbers();
  TestStrings();
  TestObjectsAndArrays();
  TestAlternatives();
  TestReferences();
  TestLimitsAndState();
  TestProofAndResourceRegressions();
  TestSharedProofBudgetAndCombinedDepth();
  TestSchemaErrors();
  std::cout << "Incremental JSON constraint checks passed.\n";
}
