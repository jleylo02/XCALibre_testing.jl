# Runtime and solvers
*Final steps before launching a simulation*

## Runtime setup
---

At this stage in the setup workflow, the user defines the runtime information to configure the runtime behaviour of the  flow solver, including the time step to use (only meaningful for transient solutions), as well as information about how often to write results to disk. XCALibre.jl provides a the `set_runtime` to perform this operation. 

```@docs; canonical=false
set_runtime
```

## Configuration object
---

Once all the simulation configuration information has been defined, from discretisation scheme to runtime information, all settings must be wrapped in a `Configuration` object. The definition, including expected input arguments, for the `Configuration` object are detailed below.

```@docs; canonical=false
Configuration
```

## Initialising fields
---

The last (optional) step before running the simulation is to provide a guess for all the fields being solved. Although this step is optional, in most cases the flow solvers will perform better when initialised. To set an initial value for a field, the `initialise!` function is provided, which assigns a starting value to a given field.

```@docs; canonical=false
initialise!
```

## Launching flow solvers
---

In XCALibre.jl the `run!` function is used to start a simulation, which will dispatch to the appropriate flow solver for execution. Once the simulation is complete a `NamedTuple` containing residual information is returned for users to explore the convergence history of the simulation. 

```@docs; canonical=false
run!()
```
---