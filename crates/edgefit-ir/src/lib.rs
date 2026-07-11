use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub type EdgeFitResult<T> = Result<T, String>;

#[derive(Clone, Debug, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    pub fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            JsonValue::Object(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            JsonValue::Number(value) if value.fract() == 0.0 => Some(*value as i64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(value) => Some(*value),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dim {
    Known(i64),
    Symbol(String),
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: Option<String>,
    pub shape: Option<Vec<Dim>>,
    pub bytes: Option<u64>,
    pub initializer: bool,
}

impl TensorInfo {
    pub fn has_static_shape(&self) -> bool {
        self.shape
            .as_ref()
            .map(|shape| {
                shape
                    .iter()
                    .all(|dim| matches!(dim, Dim::Known(value) if *value >= 0))
            })
            .unwrap_or(false)
    }

    pub fn byte_size(&self) -> Option<u64> {
        if let Some(bytes) = self.bytes {
            return Some(bytes);
        }
        self.static_shape_byte_size()
    }

    /// 基于静态 dtype 与 shape 计算稠密张量的理论字节数，不读取声明的 bytes 字段。
    pub fn static_shape_byte_size(&self) -> Option<u64> {
        let dtype_size = dtype_bytes(self.dtype.as_deref()?)?;
        let shape = self.shape.as_ref()?;
        let mut total = dtype_size;
        for dim in shape {
            match dim {
                Dim::Known(value) if *value >= 0 => total = total.checked_mul(*value as u64)?,
                _ => return None,
            }
        }
        Some(total)
    }

    /// 使用 target profile 中的符号维上界估算张量字节数，并标记是否使用了上界。
    pub fn byte_size_with_bounds(
        &self,
        symbol_bounds: &BTreeMap<String, u64>,
    ) -> Option<(u64, bool)> {
        if let Some(bytes) = self.bytes {
            return Some((bytes, false));
        }
        let dtype_size = dtype_bytes(self.dtype.as_deref()?)?;
        let shape = self.shape.as_ref()?;
        let mut total = dtype_size;
        let mut used_bound = false;
        for dim in shape {
            let value = match dim {
                Dim::Known(value) if *value >= 0 => *value as u64,
                Dim::Symbol(symbol) => {
                    used_bound = true;
                    *symbol_bounds.get(symbol)?
                }
                _ => return None,
            };
            total = total.checked_mul(value)?;
        }
        Some((total, used_bound))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FloatAttribute(u64);

impl FloatAttribute {
    pub fn from_f64(value: f64) -> Self {
        Self(value.to_bits())
    }

    pub fn as_f64(&self) -> f64 {
        f64::from_bits(self.0)
    }
}

/// ONNX 节点属性的稳定子集；Unknown 保留未建模类型，禁止静默视为兼容。
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AttributeValue {
    Int(i64),
    Float(FloatAttribute),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<FloatAttribute>),
    Strings(Vec<String>),
    Unknown { onnx_type: i64, reason: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeInfo {
    pub name: Option<String>,
    pub domain: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: BTreeMap<String, AttributeValue>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormalizedModel {
    pub path: String,
    pub sha256: String,
    pub file_bytes: u64,
    pub external_data_file_count: u64,
    pub adapter_generated: bool,
    pub opset_versions: BTreeMap<String, u64>,
    pub shape_inference_status: String,
    pub shape_inference_error: Option<String>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub tensors: BTreeMap<String, TensorInfo>,
    pub nodes: Vec<NodeInfo>,
}

impl NormalizedModel {
    pub fn initializers(&self) -> impl Iterator<Item = &TensorInfo> {
        self.tensors.values().filter(|tensor| tensor.initializer)
    }

    pub fn non_initializers(&self) -> impl Iterator<Item = &TensorInfo> {
        self.tensors.values().filter(|tensor| !tensor.initializer)
    }
}

pub fn load_normalized_model(path: impl AsRef<Path>) -> EdgeFitResult<NormalizedModel> {
    load_normalized_model_with_provenance(path, false)
}

/// 加载由 CLI 适配流程刚生成的临时 JSON；调用方必须通过原始 `.onnx` 输入建立来源。
pub fn load_cli_adapter_output(path: impl AsRef<Path>) -> EdgeFitResult<NormalizedModel> {
    load_normalized_model_with_provenance(path, true)
}

fn load_normalized_model_with_provenance(
    path: impl AsRef<Path>,
    adapter_generated: bool,
) -> EdgeFitResult<NormalizedModel> {
    let path = path.as_ref();
    if path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.eq_ignore_ascii_case("onnx"))
        .unwrap_or(false)
    {
        return Err(
            "direct .onnx loading is handled by tools/onnx-normalize before Rust analysis"
                .to_string(),
        );
    }
    let text = fs::read_to_string(path).map_err(|err| format!("failed to read model: {err}"))?;
    let root = parse_json(&text)?;
    let root = root
        .as_object()
        .ok_or("normalized model must be a JSON object")?;
    if string_field(root, "schema").as_deref() != Some("edgefit.normalized_model.v1") {
        return Err("expected schema edgefit.normalized_model.v1".to_string());
    }

    let model = object_field(root, "model")?;
    let graph = object_field(root, "graph")?;
    let normalization = root.get("normalization").and_then(JsonValue::as_object);
    // normalization 是可编辑的输入字段，只能承载元数据，不能自行取得可信适配来源。
    let shape_inference = normalization
        .and_then(|normalization| normalization.get("shape_inference"))
        .and_then(JsonValue::as_object);
    let shape_inference_status = shape_inference
        .and_then(|item| string_field(item, "status"))
        .unwrap_or_else(|| "not_recorded".to_string());
    if !matches!(
        shape_inference_status.as_str(),
        "pass" | "failed" | "not_recorded"
    ) {
        return Err(format!(
            "unsupported normalization.shape_inference.status {shape_inference_status}"
        ));
    }
    let shape_inference_error = shape_inference.and_then(|item| string_field(item, "error"));
    let opset_versions = parse_opset_versions(model, adapter_generated)?;
    let mut tensors = BTreeMap::new();

    for section in ["inputs", "values", "outputs"] {
        for item in required_array_field(graph, section)? {
            let tensor = parse_tensor(item, false, adapter_generated)?;
            merge_tensor(&mut tensors, tensor)?;
        }
    }
    for item in required_array_field(graph, "initializers")? {
        let tensor = parse_tensor(item, true, adapter_generated)?;
        merge_tensor(&mut tensors, tensor)?;
    }

    let nodes = required_array_field(graph, "nodes")?
        .iter()
        .map(parse_node)
        .collect::<EdgeFitResult<Vec<_>>>()?;
    // 节点引用但 value_info 缺失的张量必须保留为 unknown，避免分析层把缺口当作零开销。
    for name in nodes
        .iter()
        .flat_map(|node| node.inputs.iter().chain(node.outputs.iter()))
        .filter(|name| !name.is_empty())
    {
        tensors.entry(name.clone()).or_insert_with(|| TensorInfo {
            name: name.clone(),
            dtype: None,
            shape: None,
            bytes: None,
            initializer: false,
        });
    }

    let inputs = required_array_field(graph, "inputs")?
        .iter()
        .map(|item| object_field_from_value(item).and_then(|obj| required_string(obj, "name")))
        .collect::<EdgeFitResult<Vec<_>>>()?
        .into_iter()
        .filter(|name| {
            tensors
                .get(name)
                .map(|tensor| !tensor.initializer)
                .unwrap_or(true)
        })
        .collect();
    let outputs = required_array_field(graph, "outputs")?
        .iter()
        .map(|item| object_field_from_value(item).and_then(|obj| required_string(obj, "name")))
        .collect::<EdgeFitResult<Vec<_>>>()?;

    Ok(NormalizedModel {
        path: required_string(model, "path")?,
        sha256: required_string(model, "sha256")?,
        file_bytes: required_u64(model, "file_bytes")?,
        external_data_file_count: if adapter_generated {
            required_u64(model, "external_data_file_count")?
        } else {
            number_field(model, "external_data_file_count").unwrap_or(0)
        },
        adapter_generated,
        opset_versions,
        shape_inference_status,
        shape_inference_error,
        inputs,
        outputs,
        tensors,
        nodes,
    })
}

pub fn normalize_dtype(dtype: &str) -> String {
    match dtype.to_ascii_lowercase().as_str() {
        "float" | "fp32" | "tensor(float)" => "float32".to_string(),
        "fp16" | "tensor(float16)" => "float16".to_string(),
        "bf16" | "tensor(bfloat16)" => "bfloat16".to_string(),
        "tensor(int8)" => "int8".to_string(),
        "tensor(uint8)" => "uint8".to_string(),
        "tensor(int32)" => "int32".to_string(),
        "tensor(int64)" => "int64".to_string(),
        other => other.to_string(),
    }
}

pub fn dtype_bytes(dtype: &str) -> Option<u64> {
    match normalize_dtype(dtype).as_str() {
        "bool" | "int8" | "uint8" => Some(1),
        "int16" | "uint16" | "float16" | "bfloat16" => Some(2),
        "int32" | "uint32" | "float32" => Some(4),
        "int64" | "uint64" | "float64" | "double" => Some(8),
        _ => None,
    }
}

pub fn escape_json(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            value if value.is_control() => out.push_str(&format!("\\u{:04x}", value as u32)),
            value => out.push(value),
        }
    }
    out
}

pub fn parse_json(input: &str) -> EdgeFitResult<JsonValue> {
    let mut parser = Parser::new(input);
    let value = parser.parse_value()?;
    parser.skip_ws();
    if parser.pos != parser.chars.len() {
        return Err("trailing content after JSON value".to_string());
    }
    Ok(value)
}

fn parse_tensor(
    value: &JsonValue,
    initializer: bool,
    verify_declared_bytes: bool,
) -> EdgeFitResult<TensorInfo> {
    let obj = object_field_from_value(value)?;
    let tensor = TensorInfo {
        name: required_string(obj, "name")?,
        dtype: string_field(obj, "dtype"),
        shape: shape_field(obj, "shape")?,
        bytes: number_field(obj, "bytes"),
        initializer,
    };
    if verify_declared_bytes {
        if let (Some(declared), Some(expected)) = (tensor.bytes, tensor.static_shape_byte_size()) {
            if declared != expected {
                return Err(format!(
                    "declared bytes for tensor {} do not match static dtype and shape",
                    tensor.name
                ));
            }
        }
    }
    Ok(tensor)
}

fn parse_node(value: &JsonValue) -> EdgeFitResult<NodeInfo> {
    let obj = object_field_from_value(value)?;
    let domain = string_field(obj, "domain").unwrap_or_default();
    Ok(NodeInfo {
        name: string_field(obj, "name"),
        domain: if domain.is_empty() {
            "ai.onnx".to_string()
        } else {
            domain
        },
        op_type: required_string(obj, "op_type")?,
        inputs: required_string_array_field(obj, "inputs")?,
        outputs: required_string_array_field(obj, "outputs")?,
        attributes: parse_attributes(obj)?,
    })
}

fn parse_attributes(
    node: &BTreeMap<String, JsonValue>,
) -> EdgeFitResult<BTreeMap<String, AttributeValue>> {
    // v1 历史文件没有 attributes；缺失时按空集合读取，保持向后兼容。
    let Some(attributes) = node.get("attributes") else {
        return Ok(BTreeMap::new());
    };
    let attributes = attributes
        .as_object()
        .ok_or("node attributes must be a JSON object")?;
    attributes
        .iter()
        .map(|(name, value)| Ok((name.clone(), parse_attribute(value)?)))
        .collect()
}

fn parse_attribute(value: &JsonValue) -> EdgeFitResult<AttributeValue> {
    let attribute = object_field_from_value(value)?;
    let kind = required_string(attribute, "kind")?;
    match kind.as_str() {
        "int" => Ok(AttributeValue::Int(required_decimal_i64(attribute, "value")?)),
        "float" => Ok(AttributeValue::Float(FloatAttribute::from_f64(
            required_number(attribute, "value")?,
        ))),
        "string" => Ok(AttributeValue::String(required_string(attribute, "value")?)),
        "ints" => Ok(AttributeValue::Ints(required_decimal_i64_array(
            attribute, "value",
        )?)),
        "floats" => Ok(AttributeValue::Floats(
            required_number_array(attribute, "value")?
                .into_iter()
                .map(FloatAttribute::from_f64)
                .collect(),
        )),
        "strings" => Ok(AttributeValue::Strings(required_string_array_field(
            attribute, "value",
        )?)),
        "unknown" => Ok(AttributeValue::Unknown {
            onnx_type: required_i64(attribute, "onnx_type")?,
            reason: required_string(attribute, "reason")?,
        }),
        _ => Err(format!("unsupported node attribute kind {kind}")),
    }
}

fn parse_opset_versions(
    model: &BTreeMap<String, JsonValue>,
    required: bool,
) -> EdgeFitResult<BTreeMap<String, u64>> {
    let imports = match array_field(model, "opset_imports") {
        Some(imports) => imports,
        None if required => return Err("missing array field model.opset_imports".to_string()),
        None => return Ok(BTreeMap::new()),
    };
    let mut versions = BTreeMap::new();
    for item in imports {
        let item = object_field_from_value(item)?;
        let raw_domain = required_string(item, "domain")?;
        let domain = if raw_domain.is_empty() {
            "ai.onnx".to_string()
        } else {
            raw_domain
        };
        let version = required_u64(item, "version")?;
        if version == 0 {
            return Err(format!("opset version for domain {domain} must be greater than zero"));
        }
        if versions.insert(domain.clone(), version).is_some() {
            return Err(format!("duplicate opset import for domain {domain}"));
        }
    }
    Ok(versions)
}

fn merge_tensor(tensors: &mut BTreeMap<String, TensorInfo>, tensor: TensorInfo) -> EdgeFitResult<()> {
    let Some(existing) = tensors.get_mut(&tensor.name) else {
        tensors.insert(tensor.name.clone(), tensor);
        return Ok(());
    };
    if let (Some(existing_dtype), Some(new_dtype)) = (&existing.dtype, &tensor.dtype) {
        if normalize_dtype(existing_dtype) != normalize_dtype(new_dtype) {
            return Err(format!("conflicting dtype metadata for tensor {}", tensor.name));
        }
    }
    if let (Some(existing_shape), Some(new_shape)) = (&existing.shape, &tensor.shape) {
        if existing_shape != new_shape {
            return Err(format!("conflicting shape metadata for tensor {}", tensor.name));
        }
    }
    if let (Some(existing_bytes), Some(new_bytes)) = (existing.bytes, tensor.bytes) {
        if existing_bytes != new_bytes {
            return Err(format!("conflicting byte metadata for tensor {}", tensor.name));
        }
    }
    if tensor.dtype.is_some() {
        existing.dtype = tensor.dtype;
    }
    if tensor.shape.is_some() {
        existing.shape = tensor.shape;
    }
    if tensor.bytes.is_some() {
        existing.bytes = tensor.bytes;
    }
    existing.initializer |= tensor.initializer;
    Ok(())
}

fn object_field<'a>(
    obj: &'a BTreeMap<String, JsonValue>,
    key: &str,
) -> EdgeFitResult<&'a BTreeMap<String, JsonValue>> {
    obj.get(key)
        .and_then(JsonValue::as_object)
        .ok_or_else(|| format!("missing object field {key}"))
}

fn object_field_from_value(value: &JsonValue) -> EdgeFitResult<&BTreeMap<String, JsonValue>> {
    value
        .as_object()
        .ok_or_else(|| "expected object".to_string())
}

fn array_field<'a>(obj: &'a BTreeMap<String, JsonValue>, key: &str) -> Option<&'a [JsonValue]> {
    obj.get(key).and_then(JsonValue::as_array)
}

fn required_array_field<'a>(
    obj: &'a BTreeMap<String, JsonValue>,
    key: &str,
) -> EdgeFitResult<&'a [JsonValue]> {
    array_field(obj, key).ok_or_else(|| format!("missing array field {key}"))
}

fn string_field(obj: &BTreeMap<String, JsonValue>, key: &str) -> Option<String> {
    obj.get(key)
        .and_then(JsonValue::as_str)
        .map(ToString::to_string)
}

fn required_string(obj: &BTreeMap<String, JsonValue>, key: &str) -> EdgeFitResult<String> {
    string_field(obj, key).ok_or_else(|| format!("missing string field {key}"))
}

fn number_field(obj: &BTreeMap<String, JsonValue>, key: &str) -> Option<u64> {
    obj.get(key)
        .and_then(JsonValue::as_i64)
        .and_then(|value| value.try_into().ok())
}

fn required_u64(obj: &BTreeMap<String, JsonValue>, key: &str) -> EdgeFitResult<u64> {
    number_field(obj, key).ok_or_else(|| format!("missing non-negative integer field {key}"))
}

fn required_i64(obj: &BTreeMap<String, JsonValue>, key: &str) -> EdgeFitResult<i64> {
    obj.get(key)
        .and_then(JsonValue::as_i64)
        .ok_or_else(|| format!("missing integer field {key}"))
}

fn required_decimal_i64(obj: &BTreeMap<String, JsonValue>, key: &str) -> EdgeFitResult<i64> {
    required_string(obj, key)?
        .parse::<i64>()
        .map_err(|_| format!("field {key} must be a decimal int64 string"))
}

fn required_number(obj: &BTreeMap<String, JsonValue>, key: &str) -> EdgeFitResult<f64> {
    match obj.get(key) {
        Some(JsonValue::Number(value)) if value.is_finite() => Ok(*value),
        _ => Err(format!("missing finite number field {key}")),
    }
}

fn required_decimal_i64_array(
    obj: &BTreeMap<String, JsonValue>,
    key: &str,
) -> EdgeFitResult<Vec<i64>> {
    required_array_field(obj, key)?
        .iter()
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| format!("{key} must contain only decimal int64 strings"))?
                .parse::<i64>()
                .map_err(|_| format!("{key} must contain only decimal int64 strings"))
        })
        .collect()
}

fn required_number_array(
    obj: &BTreeMap<String, JsonValue>,
    key: &str,
) -> EdgeFitResult<Vec<f64>> {
    required_array_field(obj, key)?
        .iter()
        .map(|value| match value {
            JsonValue::Number(number) if number.is_finite() => Ok(*number),
            _ => Err(format!("{key} must contain only finite numbers")),
        })
        .collect()
}

fn shape_field(obj: &BTreeMap<String, JsonValue>, key: &str) -> EdgeFitResult<Option<Vec<Dim>>> {
    let Some(values) = array_field(obj, key) else {
        return Ok(None);
    };
    values
        .iter()
        .map(|value| match value {
            JsonValue::Number(number) if number.fract() == 0.0 => Ok(Dim::Known(*number as i64)),
            JsonValue::String(value) => Ok(Dim::Symbol(value.clone())),
            JsonValue::Null => Ok(Dim::Unknown),
            _ => Err("shape values must be numbers, strings, or null".to_string()),
        })
        .collect::<EdgeFitResult<Vec<_>>>()
        .map(Some)
}

fn string_array_field(obj: &BTreeMap<String, JsonValue>, key: &str) -> EdgeFitResult<Vec<String>> {
    let Some(values) = array_field(obj, key) else {
        return Ok(Vec::new());
    };
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToString::to_string)
                .ok_or_else(|| format!("{key} must contain only strings"))
        })
        .collect()
}

fn required_string_array_field(
    obj: &BTreeMap<String, JsonValue>,
    key: &str,
) -> EdgeFitResult<Vec<String>> {
    if array_field(obj, key).is_none() {
        return Err(format!("missing array field {key}"));
    }
    string_array_field(obj, key)
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn new(input: &str) -> Self {
        Self {
            chars: input.chars().collect(),
            pos: 0,
        }
    }

    fn parse_value(&mut self) -> EdgeFitResult<JsonValue> {
        self.skip_ws();
        match self.peek() {
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('"') => self.parse_string().map(JsonValue::String),
            Some('t') => self.parse_literal("true", JsonValue::Bool(true)),
            Some('f') => self.parse_literal("false", JsonValue::Bool(false)),
            Some('n') => self.parse_literal("null", JsonValue::Null),
            Some('-' | '0'..='9') => self.parse_number().map(JsonValue::Number),
            Some(value) => Err(format!("unexpected JSON character {value}")),
            None => Err("unexpected end of JSON".to_string()),
        }
    }

    fn parse_object(&mut self) -> EdgeFitResult<JsonValue> {
        self.expect('{')?;
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.consume('}') {
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(':')?;
            let value = self.parse_value()?;
            map.insert(key, value);
            self.skip_ws();
            if self.consume('}') {
                break;
            }
            self.expect(',')?;
        }
        Ok(JsonValue::Object(map))
    }

    fn parse_array(&mut self) -> EdgeFitResult<JsonValue> {
        self.expect('[')?;
        let mut values = Vec::new();
        self.skip_ws();
        if self.consume(']') {
            return Ok(JsonValue::Array(values));
        }
        loop {
            values.push(self.parse_value()?);
            self.skip_ws();
            if self.consume(']') {
                break;
            }
            self.expect(',')?;
        }
        Ok(JsonValue::Array(values))
    }

    fn parse_string(&mut self) -> EdgeFitResult<String> {
        self.expect('"')?;
        let mut out = String::new();
        while let Some(ch) = self.next() {
            match ch {
                '"' => return Ok(out),
                '\\' => {
                    let escaped = self.next().ok_or("unterminated JSON escape")?;
                    match escaped {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'b' => out.push('\u{0008}'),
                        'f' => out.push('\u{000c}'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'u' => {
                            let mut code = String::new();
                            for _ in 0..4 {
                                code.push(self.next().ok_or("truncated unicode escape")?);
                            }
                            let value = u16::from_str_radix(&code, 16)
                                .map_err(|_| "invalid unicode escape")?;
                            out.push(char::from_u32(value as u32).ok_or("invalid unicode scalar")?);
                        }
                        other => return Err(format!("invalid JSON escape {other}")),
                    }
                }
                value => out.push(value),
            }
        }
        Err("unterminated JSON string".to_string())
    }

    fn parse_number(&mut self) -> EdgeFitResult<f64> {
        let start = self.pos;
        if self.peek() == Some('-') {
            self.pos += 1;
        }
        while matches!(self.peek(), Some('0'..='9')) {
            self.pos += 1;
        }
        if self.peek() == Some('.') {
            self.pos += 1;
            while matches!(self.peek(), Some('0'..='9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some('+' | '-')) {
                self.pos += 1;
            }
            while matches!(self.peek(), Some('0'..='9')) {
                self.pos += 1;
            }
        }
        self.chars[start..self.pos]
            .iter()
            .collect::<String>()
            .parse::<f64>()
            .map_err(|err| format!("invalid JSON number: {err}"))
    }

    fn parse_literal(&mut self, literal: &str, value: JsonValue) -> EdgeFitResult<JsonValue> {
        for expected in literal.chars() {
            self.expect(expected)?;
        }
        Ok(value)
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\n' | '\r' | '\t' | '\u{feff}')) {
            self.pos += 1;
        }
    }

    fn consume(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, expected: char) -> EdgeFitResult<()> {
        match self.next() {
            Some(value) if value == expected => Ok(()),
            Some(value) => Err(format!("expected {expected}, found {value}")),
            None => Err(format!("expected {expected}, found end of input")),
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn next(&mut self) -> Option<char> {
        let value = self.peek()?;
        self.pos += 1;
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_json_object() {
        let value = parse_json(r#"{"schema":"edgefit.normalized_model.v1","n":2}"#).unwrap();
        let obj = value.as_object().unwrap();
        assert_eq!(
            obj.get("schema").and_then(JsonValue::as_str),
            Some("edgefit.normalized_model.v1")
        );
        assert_eq!(obj.get("n").and_then(JsonValue::as_i64), Some(2));
    }

    #[test]
    fn node_attributes_are_optional_for_historical_v1_models() {
        let node = parse_json(
            r#"{"name":null,"domain":"ai.onnx","op_type":"Relu","inputs":["x"],"outputs":["y"]}"#,
        )
        .unwrap();

        assert!(parse_node(&node).unwrap().attributes.is_empty());
    }

    #[test]
    fn parses_typed_and_unknown_node_attributes() {
        let node = parse_json(
            r#"{"domain":"ai.onnx","op_type":"Example","inputs":[],"outputs":[],"attributes":{"axis":{"kind":"int","value":"1"},"alpha":{"kind":"float","value":0.5},"labels":{"kind":"strings","value":["a","b"]},"tensor":{"kind":"unknown","onnx_type":4,"reason":"unmodeled_attribute_type"}}}"#,
        )
        .unwrap();
        let attributes = parse_node(&node).unwrap().attributes;

        assert_eq!(attributes.get("axis"), Some(&AttributeValue::Int(1)));
        assert_eq!(
            attributes.get("alpha"),
            Some(&AttributeValue::Float(FloatAttribute::from_f64(0.5)))
        );
        assert_eq!(
            attributes.get("labels"),
            Some(&AttributeValue::Strings(vec!["a".to_string(), "b".to_string()]))
        );
        assert_eq!(
            attributes.get("tensor"),
            Some(&AttributeValue::Unknown {
                onnx_type: 4,
                reason: "unmodeled_attribute_type".to_string(),
            })
        );
    }
}
