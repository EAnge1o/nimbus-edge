//! Host platform facts for bench and status: peak process memory, CPU model,
//! total RAM. std-only FFI against system libraries that every Windows/Linux
//! install provides; unsupported platforms report `None` honestly.

#[cfg(target_os = "windows")]
mod imp {
    use std::ffi::c_void;

    const HHKEY_LOCAL_MACHINE: usize = 0x8000_0002;
    const RRF_RT_REG_SZ: u32 = 0x2;
    const PROCESS_MEMORY_COUNTERS_SIZE: u32 = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;

    #[repr(C)]
    #[allow(non_snake_case)] // Win32 API field names
    struct PROCESS_MEMORY_COUNTERS {
        cb: u32,
        PageFaultCount: u32,
        PeakWorkingSetSize: usize,
        WorkingSetSize: usize,
        QuotaPeakPagedPoolUsage: usize,
        QuotaPagedPoolUsage: usize,
        QuotaPeakNonPagedPoolUsage: usize,
        QuotaNonPagedPoolUsage: usize,
        PagefileUsage: usize,
        PeakPagefileUsage: usize,
    }

    #[repr(C)]
    #[allow(non_snake_case)] // Win32 API field names
    #[allow(clippy::upper_case_acronyms)]
    struct MEMORYSTATUSEX {
        dwLength: u32,
        dwMemoryLoad: u32,
        ullTotalPhys: u64,
        ullAvailPhys: u64,
        ullTotalPageFile: u64,
        ullAvailPageFile: u64,
        ullTotalVirtual: u64,
        ullAvailVirtual: u64,
        ullAvailExtendedVirtual: u64,
    }

    #[link(name = "psapi")]
    extern "system" {
        fn GetProcessMemoryInfo(
            process: usize,
            counters: *mut PROCESS_MEMORY_COUNTERS,
            cb: u32,
        ) -> i32;
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> usize;
        fn GlobalMemoryStatusEx(buffer: *mut MEMORYSTATUSEX) -> i32;
    }

    #[link(name = "advapi32")]
    extern "system" {
        fn RegGetValueW(
            hkey: usize,
            sub_key: *const u16,
            value: *const u16,
            flags: u32,
            r#type: *mut u32,
            data: *mut c_void,
            len: *mut u32,
        ) -> i32;
    }

    pub fn peak_rss_bytes() -> Option<u64> {
        let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
        counters.cb = PROCESS_MEMORY_COUNTERS_SIZE;
        let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
        (ok != 0).then_some(counters.PeakWorkingSetSize as u64)
    }

    pub fn total_memory_bytes() -> Option<u64> {
        let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        (ok != 0).then_some(status.ullTotalPhys)
    }

    pub fn cpu_model() -> Option<String> {
        fn wide(s: &str) -> Vec<u16> {
            s.encode_utf16().chain(std::iter::once(0)).collect()
        }
        let sub = wide(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0");
        let value = wide("ProcessorNameString");
        let mut buf = [0u16; 128];
        let mut len = (buf.len() * 2) as u32;
        let mut kind = 0u32;
        let ok = unsafe {
            RegGetValueW(
                HHKEY_LOCAL_MACHINE,
                sub.as_ptr(),
                value.as_ptr(),
                RRF_RT_REG_SZ,
                &mut kind,
                buf.as_mut_ptr() as *mut c_void,
                &mut len,
            )
        };
        if ok != 0 {
            return None;
        }
        let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        Some(String::from_utf16_lossy(&buf[..end]).trim().to_string())
    }
}

#[cfg(target_os = "linux")]
mod imp {
    fn field(path: &str, key: &str) -> Option<u64> {
        let text = std::fs::read_to_string(path).ok()?;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix(key) {
                return rest
                    .trim_start_matches(':')
                    .split_whitespace()
                    .next()?
                    .parse()
                    .ok();
            }
        }
        None
    }

    pub fn peak_rss_bytes() -> Option<u64> {
        // VmHWM is in kB.
        field("/proc/self/status", "VmHWM").map(|kb| kb * 1024)
    }

    pub fn total_memory_bytes() -> Option<u64> {
        field("/proc/meminfo", "MemTotal").map(|kb| kb * 1024)
    }

    pub fn cpu_model() -> Option<String> {
        let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("model name") {
                return rest
                    .trim_start_matches(':')
                    .trim()
                    .trim_matches('"')
                    .to_string()
                    .into();
            }
        }
        None
    }
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
mod imp {
    pub fn peak_rss_bytes() -> Option<u64> {
        None
    }
    pub fn total_memory_bytes() -> Option<u64> {
        None
    }
    pub fn cpu_model() -> Option<String> {
        None
    }
}

pub use imp::{cpu_model, peak_rss_bytes, total_memory_bytes};

/// Build metadata captured by build.rs.
pub fn build_info() -> serde_json::Value {
    serde_json::json!({
        "rustc": env!("NIMBUS_BUILD_RUSTC"),
        "target": env!("NIMBUS_BUILD_TARGET"),
        "profile": env!("NIMBUS_BUILD_PROFILE"),
        "opt_level": if cfg!(debug_assertions) { "debug" } else { "release" },
    })
}

/// Human-readable bytes (for the bench report).
pub fn human_bytes(v: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = v as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1}{}", UNITS[unit])
}
