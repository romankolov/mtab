//! Read one onset-strength value per stdin line; print lag evidence as CSV.
use mtab_audio::{onset::OnsetEnvelope, periodicity::autocorrelation};
use std::io::{self, BufRead, Write};

fn main() -> mtab_core::Result<()> {
    let mut strength = Vec::new();
    for line in io::stdin().lock().lines() {
        let line = line?;
        let value = line.parse::<f64>().map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "expected one float per line")
        })?;
        strength.push(value);
        if strength.len() > mtab_audio::onset::MAX_ONSET_FRAMES {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "too many frames").into());
        }
    }
    let evidence = autocorrelation(&OnsetEnvelope { strength })?;
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    for row in evidence.lags {
        writeln!(
            output,
            "{},{:.17e},{:.17e},{}",
            row.lag,
            row.bpm,
            row.correlation,
            u8::from(row.local_maximum)
        )?;
    }
    Ok(())
}
