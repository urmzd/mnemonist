# Changelog

## 0.4.4 (2026-04-04)

### Documentation

- **spec**: update llmem specification to v0.2.0 (#6) ([46dcad7](https://github.com/urmzd/mnemonist/commit/46dcad73fb4ad895a10ab79a2b210e77ce9a2a9d))

### Refactoring

- **index**: replace tree-sitter with pluggable ChunkingStrategy (#7) ([1a8e562](https://github.com/urmzd/mnemonist/commit/1a8e56279fc04bf45aa0bb26c6af739438817178))

### Miscellaneous

- rename project from llmem to mnemonist ([6eb9d11](https://github.com/urmzd/mnemonist/commit/6eb9d119aec3628fdef0d6f846d68a3bac5b2e5b))

[Full Changelog](https://github.com/urmzd/mnemonist/compare/v0.4.3...v0.4.4)


## 0.4.3 (2026-04-04)

### Bug Fixes

- **cli**: canonicalize root path + add mnemonist-evals crate (#5) ([35d3d15](https://github.com/urmzd/mnemonist/commit/35d3d156e4cde47ba18efa437dedcff27f156254))

### Documentation

- **readme**: separate install commands into individual code blocks ([9993759](https://github.com/urmzd/mnemonist/commit/99937594cbb699fb9b64d15baf877c8f788dbfe8))

[Full Changelog](https://github.com/urmzd/mnemonist/compare/v0.4.2...v0.4.3)


## 0.4.2 (2026-04-02)

### Bug Fixes

- **ci**: drop musl targets from build matrix ([1cc06c8](https://github.com/urmzd/mnemonist/commit/1cc06c8c7990dc9445d8bb25aa4e7be7f8786e3d))

[Full Changelog](https://github.com/urmzd/mnemonist/compare/v0.4.1...v0.4.2)


## 0.4.1 (2026-04-01)

### Bug Fixes

- **build**: switch fastembed from native-tls to rustls-tls ([5a8978a](https://github.com/urmzd/mnemonist/commit/5a8978a205f83b45a5cab6d14d077f8d05827645))

[Full Changelog](https://github.com/urmzd/mnemonist/compare/v0.4.0...v0.4.1)


## 0.4.0 (2026-04-01)

### Features

- fastembed migration, daemon mode, TUI output, and associative linking (#4) ([3ca2c39](https://github.com/urmzd/mnemonist/commit/3ca2c392f3158abe03cd0f36d8068106dab50991))

### Miscellaneous

- update sr action from v2 to v3 ([3610251](https://github.com/urmzd/mnemonist/commit/36102512cc7dbf7fd146565d75f450df8df5714c))
- lowercase justfile with expanded recipes and add crates.io badge ([3295794](https://github.com/urmzd/mnemonist/commit/329579484197b3cf1b294e8b8a75df27172e0a67))
- standardize CI/CD — clean sr.yaml, justfile recipes, embed-src sync, teasr demo ([83dc92d](https://github.com/urmzd/mnemonist/commit/83dc92d755f23d19c8e88c964ae803056c440213))

[Full Changelog](https://github.com/urmzd/mnemonist/compare/v0.3.0...v0.4.0)


## 0.3.0 (2026-03-29)

### Features

- **validate**: add comprehensive E2E validation script ([8f27e2c](https://github.com/urmzd/mnemonist/commit/8f27e2c24012f7ece8ea4e8d08e23bd771ef05a6))

### Bug Fixes

- **test**: redact embedded field in memorize snapshot ([076c74e](https://github.com/urmzd/mnemonist/commit/076c74e0ba1f351e90916b779b069ef5403fefc9))
- **cli**: fallback to created_at when last_accessed is not set ([98e1efc](https://github.com/urmzd/mnemonist/commit/98e1efc5cb180cece7fca6f52692c9aba93de45b))

### Documentation

- add benchmark results to README via embed-src ([bd9d6b1](https://github.com/urmzd/mnemonist/commit/bd9d6b1051e68a473a3414709c9fc2ee1475e709))
- add testing guide to README and CONTRIBUTING ([a5127f2](https://github.com/urmzd/mnemonist/commit/a5127f261fb12e7a79fc481448aac03ad7d47f81))

### Refactoring

- rename mnemonist-cli package to mnemonist for simpler cargo install ([f9be434](https://github.com/urmzd/mnemonist/commit/f9be434d26a06a2bb6c4ec2cc21f18584c741b68))
- **server**: cache resolved project memory directory in app state ([9783d32](https://github.com/urmzd/mnemonist/commit/9783d326a7225d74e72854b339193576ec799181))

### Miscellaneous

- add criterion benchmarks and insta snapshot tests ([d08e712](https://github.com/urmzd/mnemonist/commit/d08e712c6779878f5812c7a390d3efc759b8ac6e))
- **cli**: add comprehensive integration tests ([22b1248](https://github.com/urmzd/mnemonist/commit/22b124859e98168611b001416334bb2b088c0917))
- **server**: extract router builder and add comprehensive tests ([e60cc66](https://github.com/urmzd/mnemonist/commit/e60cc66b3418aaae3ffaa087309bc857c2c96638))
- **deps**: add testing dependencies to server and CLI ([53bb556](https://github.com/urmzd/mnemonist/commit/53bb556d453053951cc3b7dca4d12d4ba2571e07))

[Full Changelog](https://github.com/urmzd/mnemonist/compare/v0.2.0...v0.3.0)


## 0.2.0 (2026-03-27)

### Features

- **core**: add working memory inbox module ([0d3c93f](https://github.com/urmzd/mnemonist/commit/0d3c93f95aff455fe410ef0acfea89229a4fd776))
- **core**: add memory metadata and consolidation support ([da5cf63](https://github.com/urmzd/mnemonist/commit/da5cf635b8e9e6cc83bf193e1f3b279f7c8a502a))
- **quant**: add TurboQuant vector quantization library ([8f434ed](https://github.com/urmzd/mnemonist/commit/8f434ed05ee08e0f49fae0dfd342a158b691cba1))
- **train**: add python training infrastructure ([1afbfd3](https://github.com/urmzd/mnemonist/commit/1afbfd340489abd93ee0aeb9116a36b86b38114e))
- centralized storage and config system ([25728c6](https://github.com/urmzd/mnemonist/commit/25728c6eaf266e452760246ce079e37271b0751b))
- JSON-first CLI, Ollama embedder, recall/learn/code commands ([1682010](https://github.com/urmzd/mnemonist/commit/1682010cb27596cba3dbe7224ad8a3ffb6dfc813))

### Documentation

- update architecture and api references for cognitive memory ([aa09b36](https://github.com/urmzd/mnemonist/commit/aa09b36add23abe62a1ff667802f592d377f083b))
- **readme**: document cognitive CLI and memory features ([83501d7](https://github.com/urmzd/mnemonist/commit/83501d7609565326adb1da8c4b24550337341e71))
- **spec**: specify cognitive memory with inbox and consolidation ([8291e0b](https://github.com/urmzd/mnemonist/commit/8291e0ba646480123c76d628fb59244d9d80a70e))
- **quant**: add TurboQuant vector quantization library documentation ([0b9c431](https://github.com/urmzd/mnemonist/commit/0b9c431e9db97219d8a901b5383fae4fdb439bf1))
- **skill**: add mnemonist agent skill specification ([cd5ffe5](https://github.com/urmzd/mnemonist/commit/cd5ffe5bbedc13ed7930e5af4022d571006d1a9a))
- **data**: add turboquant reference and dataset structure ([779072a](https://github.com/urmzd/mnemonist/commit/779072a305275b587e68c8ce6b753539f07784bc))
- **readme**: add install.sh as primary install method ([bcf0fe4](https://github.com/urmzd/mnemonist/commit/bcf0fe45c357282e778436f6ebaa52cb0ad013ce))
- add per-crate READMEs for crates.io ([68af45e](https://github.com/urmzd/mnemonist/commit/68af45e86b8979223bfaa54ab60d481628a3629d))
- **readme**: add config commands and configuration section ([25f0aaa](https://github.com/urmzd/mnemonist/commit/25f0aaafe5412846e18117e1ec9c3b30934d961d))

### Refactoring

- **cli**: redesign commands for cognitive memory system ([340f8ba](https://github.com/urmzd/mnemonist/commit/340f8ba231b14e85f7c299a428d1cf092c5e67eb))
- rename .ai-memory to .mnemonist ([0dbf2d2](https://github.com/urmzd/mnemonist/commit/0dbf2d231a652b5ef2bf7464a5f511bc40237d7a))

### Miscellaneous

- **skills**: rename ai-memory skill to mnemonist ([897a913](https://github.com/urmzd/mnemonist/commit/897a9130fcf69f036f3b31159c471d43389c5983))
- **workspace**: add mnemonist-quant crate and dependencies ([6730456](https://github.com/urmzd/mnemonist/commit/673045636fd4657fdc5114b1744d84b0a239c7f0))
- add install.sh for downloading release binaries ([b84bd6d](https://github.com/urmzd/mnemonist/commit/b84bd6d7f8771dfdd26930c4ec6d8434d8e4fbcb))
- add Justfile with build, install, test, and fmt commands ([bc7dc07](https://github.com/urmzd/mnemonist/commit/bc7dc07371824bcda5d696b20d80cff5247e9f87))
- **release**: migrate release workflow to use sr github action ([7bb80dc](https://github.com/urmzd/mnemonist/commit/7bb80dcbde6c8323e60af0f19b83e118838310a4))
- update memory files ([481f52f](https://github.com/urmzd/mnemonist/commit/481f52f9855e038bd6cdb0045fe7220e13f66e81))

[Full Changelog](https://github.com/urmzd/mnemonist/compare/v0.1.0...v0.2.0)


## 0.1.0 (2026-03-26)

### Features

- initial mnemonist implementation ([3a39426](https://github.com/urmzd/mnemonist/commit/3a39426315b1da4e0365e1c5a88037ff96e70e52))
