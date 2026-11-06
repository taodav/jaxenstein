# DMLab entity fixtures

`dmlab_entities.json` records entity coordinates from the six official
`assets/maps/src/nav_maze_{static,random_goal}_{01,02,03}.map` files at
[google-deepmind/lab b1db91a](https://github.com/google-deepmind/lab/tree/b1db91af5b4d2f3a24466f4632a3e5e1b0829cca).
Each entry includes its source path and SHA-256. Coordinates were extracted
directly from the source entity properties, independently of the ASCII converter.
World (x, y) coordinates map to JAXenstein cell centers via
`2 * floor(coordinate / 100) + 1.5`.

For random-goal maps, `spawn_eligible_sha256` hashes a row-major NumPy boolean
matrix. Entry [goal, start] is true when the shortest path is greater than eight
edges in the official level Lua file's `entityLayer`. Candidate order is
JAXenstein row-major order (the Lua layer is reversed vertically to align axes).
These hashes were computed from the official Lua layers, independently of
JAXenstein's geometry, to verify that reset eligibility matches the source.
