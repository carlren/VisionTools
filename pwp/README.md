# Charles Bibby's PWP Tracker

## Build:

```
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Run
```
./PWP ../data/hand.avi
```

- left click to select top-left corner of bounding box
- right click to select right-bottom corner
- `p` to start tracking