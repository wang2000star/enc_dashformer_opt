package utils

import (
	"bufio"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
)

var (
	memoryProfilingEnabled bool
	aggressiveMemoryMode   bool
	memoryProfileMu        sync.Mutex
)

// SetMemoryProfiling enables inexpensive stage-level heap/RSS telemetry.
// It is intentionally disabled by default so benchmark logging does not
// perturb the champion or optimized execution paths.
func SetMemoryProfiling(enabled bool) {
	memoryProfilingEnabled = enabled
}

// SetAggressiveMemoryMode enables explicit phase-boundary scavenging. It is
// separate from telemetry so ordinary profiling does not change execution.
func SetAggressiveMemoryMode(enabled bool) {
	aggressiveMemoryMode = enabled
}

// ReleaseMemory returns dead phase-local NTT buffers to the OS before the next
// allocation-heavy phase. Callers must clear their last references first.
func ReleaseMemory(label string) {
	if !aggressiveMemoryMode {
		return
	}
	debug.FreeOSMemory()
	ProfileMemory(label + "-released")
}

// ProfileMemory prints one machine-readable snapshot. RSS and peak RSS come
// from Linux /proc, while the remaining fields come from Go's allocator.
func ProfileMemory(label string) {
	if !memoryProfilingEnabled {
		return
	}
	memoryProfileMu.Lock()
	defer memoryProfileMu.Unlock()

	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	rssKiB, peakRSSKiB := processRSSKiB()
	fmt.Printf("[memory] label=%s rss_mib=%.1f peak_rss_mib=%.1f heap_alloc_mib=%.1f heap_inuse_mib=%.1f heap_sys_mib=%.1f heap_idle_mib=%.1f heap_released_mib=%.1f stack_mib=%.1f num_gc=%d goroutines=%d\n",
		sanitizeMemoryLabel(label), kibToMiB(rssKiB), kibToMiB(peakRSSKiB), bytesToMiB(stats.HeapAlloc), bytesToMiB(stats.HeapInuse), bytesToMiB(stats.HeapSys), bytesToMiB(stats.HeapIdle), bytesToMiB(stats.HeapReleased), bytesToMiB(stats.StackInuse), stats.NumGC, runtime.NumGoroutine())
}

func sanitizeMemoryLabel(label string) string {
	return strings.NewReplacer(" ", "-", "\t", "-", "\n", "-").Replace(label)
}

func bytesToMiB(value uint64) float64 { return float64(value) / (1024 * 1024) }
func kibToMiB(value uint64) float64   { return float64(value) / 1024 }

func processRSSKiB() (rss, peak uint64) {
	file, err := os.Open("/proc/self/status")
	if err != nil {
		return 0, 0
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 2 {
			continue
		}
		value, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			continue
		}
		switch fields[0] {
		case "VmRSS:":
			rss = value
		case "VmHWM:":
			peak = value
		}
	}
	return rss, peak
}
