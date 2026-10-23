# FINCONTAGION-X
## Financial Contagion & AI-Powered Risk Intelligence Platform

**FINCONTAGION-X** is a financial-network simulation and systemic-risk analysis platform designed to investigate how financial shocks can spread through interconnected financial institutions and businesses.

The platform combines financial stress testing, exposure-network analysis, contagion propagation, machine-learning-based early warnings, and an interactive web dashboard. It also includes a document-retrieval pipeline for financial research.

The project is built around a central question:

> How can financial shocks propagate through an interconnected ecosystem, and how can simulation and machine learning help identify vulnerable entities before distress spreads?

**Project status:** Active development. The core simulation, stress-testing outputs, ML experiments, and local FastAPI dashboard are implemented. Cloud deployment and external LLM integration remain to be configured and tested.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Financial Stress-Testing Methodology](#financial-stress-testing-methodology)
- [Contagion Simulation](#contagion-simulation)
- [Machine Learning Early-Warning System](#machine-learning-early-warning-system)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API Overview](#api-overview)
- [Validation and Reproducibility](#validation-and-reproducibility)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Author](#author)

## Overview

Financial institutions and businesses are connected through lending relationships, financial instruments, funding dependencies, and commercial exposures. A shock affecting one entity can therefore create losses for other entities and potentially trigger additional defaults.

FINCONTAGION-X represents these relationships as a financial network and evaluates the effects of configurable stress scenarios. It combines rule-based financial calculations with graph-oriented contagion analysis and machine-learning models trained on simulated scenario data.

The application is designed as a research and engineering project rather than a production banking risk system.

## Problem Statement

Traditional entity-level analysis may not capture the indirect effects of financial interconnectedness. A stressed institution can affect its counterparties, which may then transmit losses to other participants.

The platform explores several related problems:

1. **Direct financial vulnerability:** How do changes in rates, asset values, and funding availability affect an entity?
2. **Network exposure:** Which entities are connected through financial relationships?
3. **Contagion propagation:** How can losses and defaults spread across counterparties?
4. **Early warning:** Can scenario-derived financial features help identify entities likely to enter a simulated distress state?
5. **Research and explainability:** How can analysts inspect the assumptions, results, and supporting financial documents?

## Project Objectives

- Build a configurable financial ecosystem containing banks, NBFCs, funds, corporations, and suppliers.
- Model financial instruments, exposures, positions, and entity-level financial characteristics.
- Apply interest-rate, asset-price, and funding-withdrawal shocks.
- Track losses and propagation through interconnected entities.
- Develop machine-learning models for scenario-based distress classification.
- Validate model behaviour across entity-grouped splits, thresholds, and multiple random seeds.
- Expose analytical results through a FastAPI backend and a browser-based dashboard.
- Develop document retrieval and question-answering capabilities for financial research.

## Key Features

### Financial network simulation

The synthetic ecosystem represents multiple categories of financial and commercial entities. The simulation maintains structured records of entities, instruments, exposures, and financial positions.

### Integrated stress testing

Configurable scenarios examine the effects of interest-rate changes, asset-price declines, and funding withdrawals. The purpose is to distinguish direct valuation effects, credit-related losses, and liquidity pressure rather than treating every stress as the same type of loss.

### Contagion analysis

The contagion engine evaluates how losses propagate across network relationships. Results include propagation rounds, entity-level loss records, cumulative losses, and default-related status changes.

### Machine-learning early warnings

Logistic Regression and Random Forest models classify simulated entity-scenario observations into distress and non-distress categories. The workflow includes threshold analysis and multiple validation experiments.

### Interactive dashboard

The original HTML, CSS, and JavaScript dashboard presents project analytics and scenario results through a browser interface served by FastAPI.

### Financial document research

The project includes a retrieval-augmented generation (RAG) pipeline for document ingestion, chunking, retrieval, and question answering. Research functionality depends on the availability of source documents, retrieval indexes, and a configured language-model service.

## System Architecture

The application follows a modular architecture:

1. **Data layer:** Synthetic entity, instrument, exposure, position, scenario, and model-result datasets.
2. **Financial calculation layer:** Stress calculations and entity-level financial impact analysis.
3. **Network layer:** Counterparty relationships and contagion propagation.
4. **Machine-learning layer:** Feature preparation, model training, prediction, and validation.
5. **API layer:** FastAPI endpoints for analytics, scenarios, documents, research, and chat.
6. **Presentation layer:** The original browser-based dashboard.
7. **Research layer:** Document ingestion, retrieval, and LLM-assisted answers.

```text
                   Financial Scenarios
                           |
                           v
                 Stress-Testing Engine
                           |
              +------------+------------+
              |                         |
              v                         v
      Entity-Level Impacts       Exposure Network
              |                         |
              |                         v
              |                Contagion Propagation
              |                         |
              +------------+------------+
                           |
                           v
                 Processed Results
                           |
              +------------+------------+
              |                         |
              v                         v
       ML Early Warnings          FastAPI Backend
                                        |
                              +---------+---------+
                              |                   |
                              v                   v
                       HTML Dashboard       Research / Chat
```

## Financial Stress-Testing Methodology

The platform uses configurable scenario assumptions to investigate different sources of financial pressure.

### Interest-rate shock

Changes in interest rates can alter the value of interest-sensitive assets and liabilities. The magnitude of the impact depends on the exposure and the valuation assumptions used.

### Asset-price shock

A decline in asset prices reduces the value of affected holdings. The resulting impact depends on the modelled asset exposure and the scenario severity.

### Funding-withdrawal shock

Funding withdrawals create liquidity pressure. They should not automatically be treated as an accounting loss: the financial impact depends on how the entity meets the withdrawal and the assumptions about available liquidity.

### Credit-related losses

Credit stress estimates use exposure, probability of default, and loss given default. A simplified expected-credit-loss relationship is:

\[
ECL = EAD \times PD \times LGD
\]

where:

- \(EAD\) is exposure at default.
- \(PD\) is probability of default.
- \(LGD\) is loss given default.

The exact interpretation of each output depends on the implementation and the assumptions configured for the scenario.

## Contagion Simulation

The contagion workflow starts from an initial stress or default condition and evaluates subsequent effects on connected entities.

At a high level, the workflow:

1. Establishes the initial scenario and entity conditions.
2. Evaluates relevant exposure relationships.
3. Calculates losses associated with propagation rounds.
4. Updates affected entity states according to the model rules.
5. Records round-level and cumulative results.
6. Summarizes affected entities, additional defaults, and contagion losses.

The simulation is assumption-driven. Its outputs represent the behaviour of the configured model and synthetic ecosystem, not observed losses in actual financial markets.

## Machine Learning Early-Warning System

The ML component estimates whether an entity-scenario observation belongs to a simulated distress category.

### Models

- **Logistic Regression:** A linear classification baseline.
- **Random Forest:** An ensemble model capable of learning nonlinear relationships between financial features and the target label.

### Dataset design

The current scenario dataset contains 80 simulated entities evaluated under four scenario configurations, producing 320 entity-scenario rows.

The features represent simulated financial and scenario characteristics. The target labels are generated according to the project's distress definitions and simulation assumptions.

### Validation approach

The project includes experiments covering:

- Entity-grouped training and testing.
- Probability-threshold selection and sensitivity.
- Recall-oriented threshold validation.
- Cross-seed validation.
- Feature ablation and feature importance.
- Saved-model inference reproducibility.
- Evaluation on additional simulated scenarios.

Entity-grouped splitting is important because observations belonging to the same entity can share characteristics. Keeping entities separated between training and testing helps assess whether a model generalizes to entities it did not see during training.

**Important:** Performance on synthetic scenario labels does not establish predictive performance on real-world financial institutions. Metrics should always be interpreted alongside label construction, class balance, threshold selection, and the validation design.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn, Joblib |
| API | FastAPI, Uvicorn, Pydantic |
| Deep learning and retrieval | PyTorch, Sentence Transformers |
| Frontend | HTML, CSS, JavaScript |
| Document processing | pypdf |
| Data storage | CSV datasets and serialized model artifacts |
| Version control | Git, GitHub |

## Project Structure

```text
FINCONTAGION-X/
├── data/
│   ├── models/                  # Saved ML models and metadata
│   ├── raw/                     # Source financial documents
│   └── processed/               # Simulation, stress and ML outputs
├── docs/                        # Project documentation
├── src/
│   └── fincontagion/
│       ├── api.py               # FastAPI application
│       ├── static/
│       │   └── index.html       # Original browser dashboard
│       └── rag/                 # Document retrieval and generation
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

The tree above highlights the main components; it is not intended to list every project module or generated dataset.

## Installation

### Prerequisites

- Python 3.13 or a compatible Python environment.
- Git.
- A terminal such as Windows PowerShell.
- Internet access for installing dependencies and retrieving any required model weights.

### 1. Clone the repository

```bash
git clone https://github.com/ankitvectors05/FINCONTAGION-X.git
cd FINCONTAGION-X
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

Dependency versions and availability may vary by platform. Large deep-learning dependencies may require additional installation time and disk space.

## Running the Application

Start the FastAPI server from the project root with the virtual environment activated:

```powershell
python -m uvicorn fincontagion.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Open the dashboard:

http://127.0.0.1:8000/dashboard

Check the API health endpoint:

http://127.0.0.1:8000/health

Keep the terminal running while using the application. Stop the server with `Ctrl+C`.

Some research and chatbot functions may require a locally running model service and a configured retrieval index. A successful dashboard launch does not by itself confirm that every research or LLM feature is available.

## API Overview

The application includes routes for the following functions:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check application health |
| `GET /api/network` | Retrieve network data |
| `GET /api/analytics` | Retrieve analytical summaries |
| `GET /api/contagion/scenarios` | Retrieve contagion scenario results |
| `POST /api/documents/upload` | Upload a document for processing |
| `POST /api/documents/{document_id}/ask` | Ask a question about an uploaded document |
| `POST /api/research/ask` | Submit a financial research query |
| `POST /api/chat` | Submit a chatbot query |

Request and response schemas should be checked in the implementation before integrating external clients. Document uploads and research requests may depend on local files or model services.

## Validation and Reproducibility

The project contains validation workflows for financial outputs and ML experiments. These include entity reconciliation, stress-result checks, threshold sensitivity, grouped evaluation, cross-seed comparisons, feature ablation, and saved-model prediction checks.

The saved-model reproducibility experiment has checked that predictions from persisted Logistic Regression and Random Forest models match the corresponding original inference outputs on the tested dataset.

Validation results should be interpreted within the boundaries of the simulated data and the exact experiment configuration. Reproducibility on a fixed dataset is not equivalent to validation on external financial data.

## Limitations

- The financial ecosystem and scenario labels are simulated.
- Model outputs depend on the selected assumptions, feature construction, and distress definitions.
- Contagion dynamics are simplified representations of complex financial relationships.
- Real-world calibration would require reliable external financial data and domain validation.
- Document research depends on available source documents, retrieval indexes, and model configuration.
- Local LLM services do not automatically become available when the application is hosted in the cloud.
- The project is not yet a production-grade, regulated financial risk-management system.

## Roadmap

Planned development areas include:

- Deploy the FastAPI backend and original HTML dashboard.
- Configure an externally accessible inference service for cloud chatbot functionality.
- Validate document ingestion and RAG retrieval in the deployment environment.
- Improve scenario configuration, auditability, and result explanations.
- Expand testing and monitoring for API and model behaviour.
- Document deployment procedures and environment configuration.

## Author

**Ankit Kumar**

GitHub: [@ankitvectors05](https://github.com/ankitvectors05)

Repository: [FINCONTAGION-X](https://github.com/ankitvectors05/FINCONTAGION-X)

## Disclaimer

FINCONTAGION-X is an educational and research-oriented software project. It does not provide financial advice, guarantee real-world risk predictions, or replace professional financial, regulatory, or risk-management systems.