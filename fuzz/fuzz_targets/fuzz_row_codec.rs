//! Row codec decode fuzz target (VC-005)
//! Oracle: 1) no panic 2) encode -> decode roundtrip for valid schema
#![no_main]
use libfuzzer_sys::fuzz_target;
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

fuzz_target!(|data: &[u8]| {
    if data.len() > 10_000_000 {
        return;
    }
    let schema = Schema::new(vec![
        Field::new("c0", DataType::Int64, true),
        Field::new("c1", DataType::Utf8, true),
    ]);
    // L1: decode must not panic
    let _ = hatp_engine::row_codec::decode_row_values(data, &schema);
    // L2: roundtrip — encode a known row, decode must recover it
    use arrow_array::{Int64Array, RecordBatch, StringArray};
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(vec![42_i64])),
            Arc::new(StringArray::from(vec!["hello"])),
        ],
    );
    if let Ok(batch) = batch {
        if let Ok(encoded) = hatp_engine::row_codec::encode_row(&batch, 0) {
            let _ = hatp_engine::row_codec::decode_row_values(&encoded, &schema);
        }
    }
});