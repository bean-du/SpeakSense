use anyhow::Result;
use chrono::Local;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::{non_blocking, rolling};
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::{fmt, layer::SubscriberExt, EnvFilter};

pub fn init(dir: String) -> Result<WorkerGuard> {
    // output to stdout
    let formatting_layer = fmt::layer()
        .pretty()
        .with_timer(LocalTimer::default())
        .with_writer(std::io::stdout);

    // output to file
    let file_appender = rolling::hourly(dir, "asr");
    let (non_blocking, _guard) = non_blocking(file_appender);

    let file_layer = fmt::layer()
        .pretty()
        .with_timer(LocalTimer::default())
        .with_ansi(false)
        .with_writer(non_blocking);

    let collector = tracing_subscriber::registry()
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .with(formatting_layer)
        .with(file_layer);

        
    tracing::subscriber::set_global_default(collector).expect("setting default subscriber failed");

    Ok(_guard)
}

#[derive(Default)]
struct LocalTimer;

impl FormatTime for LocalTimer {
    fn format_time(&self, w: &mut fmt::format::Writer<'_>) -> std::fmt::Result {
        write!(w, "{}", Local::now().format("%Y-%m-%d %H:%M"))
    }
}
