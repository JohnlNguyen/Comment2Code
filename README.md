# Comment2Code
Inline comment and Code Project

# Quick Start
Build Docker Container
```shell script
docker build -t johnlnguyen/comment2code:latest -f Dockerfile .
```

Run Docker Container
```shell script
docker run -v $(pwd):/ds -it johnlnguyen/comment2code:latest
```

Inspect Data
```shell script
make inspect
```

Randomly Shuffle and Inspect Data
```shell script
make inspect-rand
```
