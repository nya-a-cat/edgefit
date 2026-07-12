use std::collections::BTreeMap;

use crate::{Error, Result};

pub(crate) const MAX_JSON_INPUT_BYTES: usize = 16 * 1024 * 1024;
pub(crate) const MAX_JSON_DEPTH: usize = 128;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum JsonValue {
    Null,
    Bool(bool),
    Number,
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

pub(crate) struct JsonParser<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> JsonParser<'a> {
    pub(crate) fn new(input: &'a str) -> Self {
        Self {
            bytes: input.as_bytes(),
            position: 0,
        }
    }

    pub(crate) fn parse(mut self) -> Result<JsonValue> {
        if self.bytes.len() > MAX_JSON_INPUT_BYTES {
            return Err(Error::new(format!(
                "JSON input exceeds byte limit {MAX_JSON_INPUT_BYTES}"
            )));
        }
        let value = self.value(0)?;
        self.whitespace();
        if self.position != self.bytes.len() {
            return Err(Error::new("trailing content after JSON value"));
        }
        Ok(value)
    }

    fn value(&mut self, depth: usize) -> Result<JsonValue> {
        if depth > MAX_JSON_DEPTH {
            return Err(Error::new(format!(
                "JSON nesting exceeds depth limit {MAX_JSON_DEPTH}"
            )));
        }
        self.whitespace();
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => self.string().map(JsonValue::String),
            Some(b't') => self.literal(b"true", JsonValue::Bool(true)),
            Some(b'f') => self.literal(b"false", JsonValue::Bool(false)),
            Some(b'n') => self.literal(b"null", JsonValue::Null),
            Some(b'-' | b'0'..=b'9') => {
                self.number()?;
                Ok(JsonValue::Number)
            }
            Some(byte) => Err(Error::new(format!("unexpected JSON byte 0x{byte:02x}"))),
            None => Err(Error::new("unexpected end of JSON")),
        }
    }

    fn object(&mut self, depth: usize) -> Result<JsonValue> {
        self.expect(b'{')?;
        self.whitespace();
        let mut fields = BTreeMap::new();
        if self.consume(b'}') {
            return Ok(JsonValue::Object(fields));
        }
        loop {
            self.whitespace();
            let key = self.string()?;
            self.whitespace();
            self.expect(b':')?;
            let value = self.value(depth + 1)?;
            if fields.insert(key.clone(), value).is_some() {
                return Err(Error::new(format!("duplicate JSON key {key}")));
            }
            self.whitespace();
            if self.consume(b'}') {
                break;
            }
            self.expect(b',')?;
        }
        Ok(JsonValue::Object(fields))
    }

    fn array(&mut self, depth: usize) -> Result<JsonValue> {
        self.expect(b'[')?;
        self.whitespace();
        let mut values = Vec::new();
        if self.consume(b']') {
            return Ok(JsonValue::Array(values));
        }
        loop {
            values.push(self.value(depth + 1)?);
            self.whitespace();
            if self.consume(b']') {
                break;
            }
            self.expect(b',')?;
        }
        Ok(JsonValue::Array(values))
    }

    fn string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut out = String::new();
        let mut segment = self.position;
        loop {
            let byte = self
                .next()
                .ok_or_else(|| Error::new("unterminated JSON string"))?;
            match byte {
                b'"' => {
                    self.push_utf8_segment(&mut out, segment, self.position - 1)?;
                    return Ok(out);
                }
                b'\\' => {
                    self.push_utf8_segment(&mut out, segment, self.position - 1)?;
                    let escaped = self
                        .next()
                        .ok_or_else(|| Error::new("unterminated JSON escape"))?;
                    match escaped {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => out.push(self.unicode_escape()?),
                        _ => return Err(Error::new("invalid JSON escape")),
                    }
                    segment = self.position;
                }
                0x00..=0x1f => {
                    return Err(Error::new("unescaped control byte in JSON string"));
                }
                _ => {}
            }
        }
    }

    fn unicode_escape(&mut self) -> Result<char> {
        let high = self.hex_quad()?;
        let scalar = if (0xd800..=0xdbff).contains(&high) {
            self.expect(b'\\')?;
            self.expect(b'u')?;
            let low = self.hex_quad()?;
            if !(0xdc00..=0xdfff).contains(&low) {
                return Err(Error::new("invalid JSON surrogate pair"));
            }
            0x10000 + (((high - 0xd800) as u32) << 10) + (low - 0xdc00) as u32
        } else if (0xdc00..=0xdfff).contains(&high) {
            return Err(Error::new("unpaired low JSON surrogate"));
        } else {
            high as u32
        };
        char::from_u32(scalar).ok_or_else(|| Error::new("invalid Unicode scalar"))
    }

    fn push_utf8_segment(&self, out: &mut String, start: usize, end: usize) -> Result<()> {
        out.push_str(
            std::str::from_utf8(&self.bytes[start..end])
                .map_err(|_| Error::new("invalid UTF-8 in JSON string"))?,
        );
        Ok(())
    }

    fn hex_quad(&mut self) -> Result<u16> {
        let end = self
            .position
            .checked_add(4)
            .ok_or_else(|| Error::new("JSON position overflow"))?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| Error::new("truncated JSON Unicode escape"))?;
        self.position = end;
        let text =
            std::str::from_utf8(bytes).map_err(|_| Error::new("invalid Unicode escape"))?;
        u16::from_str_radix(text, 16).map_err(|_| Error::new("invalid Unicode escape"))
    }

    fn number(&mut self) -> Result<()> {
        self.consume(b'-');
        match self.peek() {
            Some(b'0') => {
                self.position += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(Error::new("leading zero in JSON number"));
                }
            }
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.position += 1;
                }
            }
            _ => return Err(Error::new("invalid JSON number")),
        }
        if self.consume(b'.') {
            let before = self.position;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.position += 1;
            }
            if before == self.position {
                return Err(Error::new("missing fraction digits"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.position += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.position += 1;
            }
            let before = self.position;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.position += 1;
            }
            if before == self.position {
                return Err(Error::new("missing exponent digits"));
            }
        }
        Ok(())
    }

    fn literal(&mut self, literal: &[u8], value: JsonValue) -> Result<JsonValue> {
        for byte in literal {
            self.expect(*byte)?;
        }
        Ok(value)
    }

    fn whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.position += 1;
        }
    }

    fn expect(&mut self, expected: u8) -> Result<()> {
        match self.next() {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(Error::new(format!(
                "expected JSON byte 0x{expected:02x}, found 0x{actual:02x}"
            ))),
            None => Err(Error::new("unexpected end of JSON")),
        }
    }

    fn consume(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let value = self.peek()?;
        self.position += 1;
        Some(value)
    }
}

pub(crate) fn object<'a>(
    value: &'a JsonValue,
    context: &str,
) -> Result<&'a BTreeMap<String, JsonValue>> {
    match value {
        JsonValue::Object(value) => Ok(value),
        _ => Err(Error::new(format!("{context} must be an object"))),
    }
}

pub(crate) fn array<'a>(value: &'a JsonValue, context: &str) -> Result<&'a [JsonValue]> {
    match value {
        JsonValue::Array(value) => Ok(value),
        _ => Err(Error::new(format!("{context} must be an array"))),
    }
}

pub(crate) fn required<'a>(
    object: &'a BTreeMap<String, JsonValue>,
    key: &str,
) -> Result<&'a JsonValue> {
    object
        .get(key)
        .ok_or_else(|| Error::new(format!("missing required field {key}")))
}

pub(crate) fn exact_fields(
    object: &BTreeMap<String, JsonValue>,
    expected: &[&str],
    context: &str,
) -> Result<()> {
    reject_forbidden_security_fields(object, context)?;
    for key in object.keys() {
        if !expected.contains(&key.as_str()) {
            return Err(Error::new(format!("unknown field {context}.{key}")));
        }
    }
    for key in expected {
        if !object.contains_key(*key) {
            return Err(Error::new(format!(
                "missing required field {context}.{key}"
            )));
        }
    }
    Ok(())
}

fn reject_forbidden_security_fields(
    object: &BTreeMap<String, JsonValue>,
    context: &str,
) -> Result<()> {
    for key in object.keys() {
        if matches!(
            key.as_str(),
            "signature" | "signatures" | "certificate" | "public_key"
        ) {
            return Err(Error::new(format!(
                "unsupported signature field {context}.{key}; v1 is hash-only"
            )));
        }
        if key == "attestation" && context != "evidence" {
            return Err(Error::new(format!(
                "unsupported attestation field {context}.{key}; v1 has no attestation"
            )));
        }
    }
    Ok(())
}

pub(crate) fn string(
    object: &BTreeMap<String, JsonValue>,
    key: &str,
) -> Result<String> {
    match required(object, key)? {
        JsonValue::String(value) => Ok(value.clone()),
        _ => Err(Error::new(format!("field {key} must be a string"))),
    }
}

pub(crate) fn nonempty_string(
    object: &BTreeMap<String, JsonValue>,
    key: &str,
) -> Result<String> {
    let value = string(object, key)?;
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(Error::new(format!(
            "field {key} must be a non-empty safe string"
        )));
    }
    Ok(value)
}

pub(crate) fn optional_string(
    object: &BTreeMap<String, JsonValue>,
    key: &str,
) -> Result<Option<String>> {
    match required(object, key)? {
        JsonValue::Null => Ok(None),
        JsonValue::String(value) => Ok(Some(value.clone())),
        _ => Err(Error::new(format!(
            "field {key} must be a string or null"
        ))),
    }
}

pub(crate) fn boolean(
    object: &BTreeMap<String, JsonValue>,
    key: &str,
) -> Result<bool> {
    match required(object, key)? {
        JsonValue::Bool(value) => Ok(*value),
        _ => Err(Error::new(format!("field {key} must be a boolean"))),
    }
}

pub(crate) fn decimal_u64(
    object: &BTreeMap<String, JsonValue>,
    key: &str,
) -> Result<u64> {
    parse_decimal_u64(&string(object, key)?, key)
}

pub(crate) fn decimal_u64_array(
    object: &BTreeMap<String, JsonValue>,
    key: &str,
) -> Result<Vec<u64>> {
    array(required(object, key)?, key)?
        .iter()
        .map(|value| match value {
            JsonValue::String(value) => parse_decimal_u64(value, key),
            _ => Err(Error::new(format!(
                "field {key} must contain decimal strings"
            ))),
        })
        .collect()
}

fn parse_decimal_u64(value: &str, key: &str) -> Result<u64> {
    if value.is_empty()
        || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(Error::new(format!(
            "field {key} must be a canonical decimal u64 string"
        )));
    }
    value
        .parse::<u64>()
        .map_err(|_| Error::new(format!("field {key} is outside the u64 range")))
}

pub(crate) fn expect_literal_string(
    object: &BTreeMap<String, JsonValue>,
    key: &str,
    expected: &str,
) -> Result<()> {
    if string(object, key)? != expected {
        return Err(Error::new(format!("field {key} must equal {expected}")));
    }
    Ok(())
}
