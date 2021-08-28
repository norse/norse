
# Quickstart

This page walks you through the initial steps to becoming productive with Norse.
We will cover how to 

* Work with neuron state
* Work with Norse without time
* Work with Norse with recurrence
* Work with Norse with time

```{code-cell} ipython3

    import torch
    import norse

    cell = norse.torch.LIFCell()
    data = torch.ones(1)
    spikes, state = cell(data)
```

The *next* time you call the cell, you need to pass in that state. 
Otherwise you will get the exact same output
