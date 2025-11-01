// Common utilities for modes

use crate::config::default::Config;

pub fn prepare_config_for_mode(base_config: &Config, custom_prompt: &str) -> Config {
    let mut config = base_config.clone();
    config.prompt = custom_prompt.to_string();
    config
}