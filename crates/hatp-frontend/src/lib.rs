//! HatP frontend — DataFusion integration, catalog, DDL, planner hooks.
//!
//! The frontend is the *logical* half of HatP. It owns the SQL surface
//! (catalog, DDL, planning) while the OLTP row data lives in
//! [`hatp_engine`]. DataFusion itself drives query execution; Vortex
//! powers the OLAP file layout behind the [`hatp_engine`]-facing
//! `TableProvider` adapter.

#![doc(html_root_url = "https://docs.rs/hatp-frontend/0.1.0")]

pub mod catalog;
pub mod dml_planner;
pub mod execution;
pub mod predicate_translator;
pub mod schema;
