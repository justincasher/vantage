# Project Structure

```
.
├── LICENSE
├── README.md
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── scripts
│   ├── test_latex_processor_basic.py
│   ├── test_lean_processor_basic.py
│   └── test_lean_processor_tree.py
├── src
│   ├── lean_automator
│   │   ├── init.py
│   │   ├── kb_search.py
│   │   ├── kb_storage.py
│   │   ├── latex_processor.py
│   │   ├── lean_interaction.py
│   │   ├── lean_processor.py
│   │   ├── lean_proof_repair.py
│   │   └── llm_call.py
│   └── vantage.egg-info
│       ├── PKG-INFO
│       ├── SOURCES.txt
│       ├── dependency_links.txt
│       └── top_level.txt
├── test_lean_proc_list_kb.sqlite
├── test_lean_proc_tree_kb.sqlite
└── tests
├── init.py
├── integration
│   ├── init.py
│   ├── test_kb_search_integration.py
│   ├── test_kb_storage_integration.py
│   ├── test_lean_interaction_integration.py
│   └── test_llm_call_integration.py
└── unit
├── init.py
├── test_kb_search_unit.py
├── test_kb_storage_unit.py
├── test_lean_interaction_unit.py
└── test_llm_call_unit.py
```

*(Note: `.sqlite` database files and `.env` are typically generated/created in the root but omitted from the source structure example)*