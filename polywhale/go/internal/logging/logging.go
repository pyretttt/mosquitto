package logging

import (
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// The Rust version logs through tui-logger to a temp file with simple size
// based rotation, and exposes the records to an in-app log overlay. Here a
// Memory sink keeps the recent lines for the overlay while the same records
// stream to the file.

const (
	logFileMaxSizeBytes = 1024 * 1024
	memoryMaxLines      = 2000
)

// Memory is an io.Writer that retains the most recent log lines.
type Memory struct {
	mu    sync.Mutex
	lines []string
}

func (m *Memory) Write(p []byte) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, line := range strings.Split(strings.TrimRight(string(p), "\n"), "\n") {
		m.lines = append(m.lines, line)
	}
	if overflow := len(m.lines) - memoryMaxLines; overflow > 0 {
		m.lines = m.lines[overflow:]
	}
	return len(p), nil
}

// Lines returns a snapshot of the retained log lines.
func (m *Memory) Lines() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]string(nil), m.lines...)
}

// Setup wires the standard logger to a rotating temp file plus the in-memory
// sink and returns the sink and the log file path.
func Setup() (*Memory, string, error) {
	path := filepath.Join(os.TempDir(), "polywhale.log")

	// Simple log rotation by file size.
	if info, err := os.Stat(path); err == nil && info.Size() >= logFileMaxSizeBytes {
		_ = os.Remove(path)
	}

	file, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, "", err
	}

	memory := &Memory{}
	log.SetFlags(0)
	log.SetOutput(io.MultiWriter(file, memory))

	Infof("app", "Logging to: %s", path)
	return memory, path, nil
}

func Infof(target, format string, args ...any) {
	write("I", target, format, args...)
}

func Errorf(target, format string, args ...any) {
	write("E", target, format, args...)
}

// write mimics tui-logger's abbreviated output: TIME:LEVEL:TARGET:message.
func write(level, target, format string, args ...any) {
	log.Printf("%s:%s:%s:%s", time.Now().Format("15:04:05"), level, target, fmt.Sprintf(format, args...))
}
