//! `mtab` CLI — scaffold. The real interface is specified in `PROMPT.md` §6.12
//! (`transcribe`, `inspect-weights`, `download`, `detect-tuning`, `eval`)
//! and lands across milestones M1–M8.

use mtab_core::ModelSize;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("mtab {}", env!("CARGO_PKG_VERSION"));
        return;
    }

    if args.iter().any(|a| a == "--models") {
        println!("MuScriptor model variants (PROMPT.md §2):");
        for size in ModelSize::ALL {
            let (params, layers, dim) = size.shape();
            println!(
                "  {:<6} ~{:<4}M params, {:>2} layers, dim {:>4}",
                format!("{:?}", size).to_lowercase(),
                params / 1_000_000,
                layers,
                dim
            );
        }
        return;
    }

    println!(
        "mtab {} — scaffold build. The real CLI arrives in milestones M1-M8; see PROMPT.md.",
        env!("CARGO_PKG_VERSION")
    );
    println!("Usage coming up: mtab transcribe song.wav --size medium --weights model.safetensors");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_sizes_are_listed() {
        assert_eq!(ModelSize::ALL.len(), 3);
    }
}
