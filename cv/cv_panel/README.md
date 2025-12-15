## Current value signal

`signals.py` provides a light-weight wrapper that exposes a `CurrentValueSignal`
as a `QObject` with a Qt `valueChanged` signal. Widgets (or plain Python
callbacks) can connect to the signal like any other PySide6 signal/slot, while
the helper tracks and exposes the latest value.

```python
from PySide6.QtCore import Slot
from signals import CurrentValueSignal

count = CurrentValueSignal(0)
double_count = count.map(lambda value: value * 2)

@Slot(object)
def handle_count(value):
    print(f"Count changed to {value}")

count.valueChanged.connect(handle_count)
double_count.subscribe(lambda value: print(f"Double is {value}"))

count.set(2)   # -> "Count changed to 2", "Double is 4"
count.update(lambda value: value + 3)  # -> "Count changed to 5", "Double is 10"
```

Signals can be bound directly to widgets. See `app.py` for a working example
that keeps a label synchronized with the count and a derived signal showing the
double of the count.
