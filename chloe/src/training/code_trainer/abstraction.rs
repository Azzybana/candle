// Handles abstraction of Rust code into training-friendly representations
// Using syn for AST parsing and extracting abstract code sections

use anyhow::Result;
use syn::{File, Item, visit::Visit};

pub fn abstract_rust_code(code: &str) -> Result<Vec<String>> {
    let syntax_tree: File = syn::parse_str(code)?;
    let mut extractor = AbstractionExtractor::new();
    extractor.visit_file(&syntax_tree);
    Ok(extractor.abstractions)
}

struct AbstractionExtractor {
    abstractions: Vec<String>,
}

impl AbstractionExtractor {
    fn new() -> Self {
        Self {
            abstractions: Vec::new(),
        }
    }
}

impl<'ast> Visit<'ast> for AbstractionExtractor {
    fn visit_item(&mut self, node: &'ast Item) {
        match node {
            Item::Fn(item_fn) => {
                self.abstractions.push(format!(
                    "function {} with {} parameters",
                    item_fn.sig.ident,
                    item_fn.sig.inputs.len()
                ));
            }
            Item::Struct(item_struct) => {
                self.abstractions.push(format!(
                    "struct {} with {} fields",
                    item_struct.ident,
                    item_struct.fields.len()
                ));
            }
            Item::Impl(item_impl) => {
                if item_impl.trait_.is_some() {
                    self.abstractions.push(format!(
                        "impl trait for {}",
                        format_type(&item_impl.self_ty)
                    ));
                } else {
                    self.abstractions.push(format!(
                        "impl for {}",
                        format_type(&item_impl.self_ty)
                    ));
                }
            }
            _ => {}
        }
        syn::visit::visit_item(self, node);
    }
}

fn format_type(ty: &syn::Type) -> String {
    // Simple type formatting
    match ty {
        syn::Type::Path(type_path) => {
            type_path.path.segments.last().map(|seg| seg.ident.to_string()).unwrap_or("unknown".to_string())
        }
        _ => "complex".to_string(),
    }
}