# graphflowDB

## Compilation

### Bazel configuration

- Create bazel configuration file with `touch .bazelrc`
- Add bazel configuration `echo build --cxxopt="-std=c++2a" --cxxopt='-O3' --cxxopt='-fopenmp' --linkopt='-lgomp' > .bazelrc`

### Bazel build

- To do a full clean build
    - `bazel clean`
    - `bazel build //...:all`
- To test
    - `bazel test //...:all`

## Dataset loading

- Dataset folder should contain a metadata.json with the following format

```
{
    "tokenSeparator": ",",
    "nodeFileDescriptions": [
        {
            "filename": "vPerson.csv",
            "label": "person",
            "primaryKey": "id"
        }
    ],
    "relFileDescriptions": [
        {
            "filename": "eKnows.csv",
            "label": "knows",
            "multiplicity": "MANY_MANY",
            "srcNodeLabels": [
                "person"
            ],
            "dstNodeLabels": [
                "person"
            ]
        },
    ]
}
```

- To serialize dataset `bazel run //src/loader:loader_runner <input-csv-path> <output-serialized-graph-path>`

## CLI

- To start CLI `bazel run //tools/shell:graphflowdb -- -i <serialized-graph-path>`
- CLI built in commands
  :help get command list
  :clear clear shell
  :quit exit from shell
  :thread [thread]     number of threads for execution
  :bm_debug_info debug information about the buffer manager
  :system_debug_info debug information about the system

## Benchmark
### Reproducing CIDR results
Please use `main` under `examples` to reproduce the results in CIDR paper. The following command will run all the queries in the paper.
```shell
./bazel-bin/examples/main <serializedPath> <queryPath> <encodedJoin> <numThreads>
```

For example, to run IS1 with 8 threads, use the following command.
```shell
./bazel-bin/examples/main serialized/ examples/queries/IS01.cypher is01 8
```

### Benchmark runner
Benchmark runner is designed to be used in graphflowdb-benchmark. Deirectly using benchmark runner is not recommended.

- benchmark file should have the following format

```
name example

query
MATCH (:Person)
RETURN COUNT(*)

expectedNumOutput 8

```

- To run benchmark
  runner `<benchmark-tool-binary-path> --dataset=<serialized-graph-path> --benchmark=<benchmark-file-or-folder>`
    - `--warmup=1` config number of warmup
    - `--run=5` config number of run
    - `--thread=1` config number of thread for execution

## Python Driver

Current python driver is under development. See `tools/python_api/test/example.py` for example usage. To run this
example `bazel run //tools/python_api:example`.
