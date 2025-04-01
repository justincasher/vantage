├── LICENSE
├── README.md
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── scripts
│   ├── test_latex_processor_basic.py
│   ├── test_lean_processor_basic.py
│   └── test_lean_processor_tree.py
├── src
│   ├── lean_automator
│   │   ├── __init__.py
│   │   ├── kb_search.py
│   │   ├── kb_storage.py
│   │   ├── latex_processor.py
│   │   ├── lean_interaction.py
│   │   ├── lean_processor.py
│   │   ├── lean_proof_repair.py
│   │   └── llm_call.py
│   └── vantage.egg-info
│       ├── PKG-INFO
│       ├── SOURCES.txt
│       ├── dependency_links.txt
│       └── top_level.txt
├── tests
│   ├── __init__.py
│   ├── integration
│   │   ├── __init__.py
│   │   ├── test_kb_search_integration.py
│   │   ├── test_kb_storage_integration.py
│   │   ├── test_lean_interaction_integration.py
│   │   └── test_llm_call_integration.py
│   └── unit
│       ├── __init__.py
│       ├── test_kb_search_unit.py
│       ├── test_kb_storage_unit.py
│       ├── test_lean_interaction_unit.py
│       └── test_llm_call_unit.py
└── vantage_lib
    ├── VantageLib
    ├── lake-manifest.json
    ├── lakefile.toml
    └── lean-toolchain
