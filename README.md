# MedLedger AI - Masumi Agents

A comprehensive medical AI platform featuring two specialized Masumi agents for medical record analysis and appointment scheduling. Built with CrewAI, Azure OpenAI, and integrated with the Masumi Network for decentralized payments and identity.

## 🏥 Overview

MedLedger AI provides two powerful AI agents that work together to deliver a complete medical service experience:

1. **Explain Agent** - Analyzes medical records, explains findings in plain language, and predicts disease risks
2. **Appointment Agent** - Finds hospitals by specialty and schedules appointments based on patient needs

Both agents are MIP-003 compliant and integrated with the Masumi Network for blockchain-based payments and identity management.

## 🤖 Agents

### 1. Explain Agent (Medical Record Analysis)

**Port:** `8000`  
**Swagger UI:** http://127.0.0.1:8000/docs

#### Features

- **Medical Record Analysis**
  - Extracts and analyzes text from PDF medical reports
  - Provides clear, patient-friendly explanations
  - Identifies key findings and abnormalities
  - Generates confidence estimates

- **Early Disease Risk Prediction** 🆕
  - Predicts risks for Type 2 Diabetes, Hypertension, and Heart Disease
  - Calculates risk scores (0.0 to 1.0) with confidence levels
  - Identifies early warning signs before diseases develop
  - Provides preventive recommendations

- **Anti-Hallucination Protection** 🆕
  - Rule-based validation against medical reference ranges
  - Requires explicit evidence quotes from records
  - Validates predictions against hard-coded medical facts
  - Prevents false positives for normal values

- **Input/Output Guardrails** 🆕
  - Blocks malicious inputs (code injection, prompt injection)
  - Validates medical appropriateness of outputs
  - Adds proper medical disclaimers automatically
  - Protects against spam and harmful content

- **Doctor-in-the-Loop Validation** 🆕
  - Flags high-risk predictions for doctor review
  - Manages validation workflow (pending/approved/rejected)
  - Tracks doctor decisions and comments
  - Ensures critical findings are reviewed by humans

#### Key Components

- `MedicalGuardrails` - Input/output safety checks
- `DiseasePredictor` - LLM-powered risk assessment with anti-hallucination
- `MedicalRules` - Hard-coded medical reference ranges and validation
- `DoctorValidationHandler` - Manages doctor review workflow

---

### 2. Appointment Agent (Hospital Search & Scheduling)

**Port:** `8001`  
**Swagger UI:** http://127.0.0.1:8001/docs

#### Features

- **Intelligent Specialty Detection**
  - Understands natural language patient requests
  - Extracts medical specialty from patient descriptions
  - Provides reasoning for specialty selection

- **Real-time Hospital Search**
  - Searches hospitals by pincode using Serper API
  - Filters by medical specialty
  - Retrieves hospital contact information (phone, address)
  - Gets ratings and availability information

- **Smart Hospital Selection**
  - LLM-powered hospital selection with reasoning
  - Considers proximity, specialty match, and ratings
  - Explains why a specific hospital was chosen

- **Appointment Scheduling**
  - Generates appointment details (date, time, confirmation number)
  - Provides hospital contact information
  - Creates structured appointment records

#### Key Components

- `AppointmentCrew` - CrewAI orchestration for appointment workflow
- `HospitalSearch` - Serper API integration for real-time search
- `HospitalBooking` - Appointment scheduling logic

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Azure OpenAI API key and endpoint
- Serper API key (for Appointment Agent)
- Masumi Node (optional, for payment integration)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Anajrajeev/MedLedger-AI.git
   cd MedLedger-AI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

### Environment Variables

Create a `.env` file with the following:

```env
# Azure OpenAI Configuration (Required)
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=your_deployment_name

# Serper API (Required for Appointment Agent)
SERPER_API_KEY=your_serper_api_key

# Masumi Payment Configuration (Optional)
PAYMENT_SERVICE_URL=http://localhost:3001
PAYMENT_API_KEY=your_masumi_api_key
AGENT_IDENTIFIER=your_agent_identifier
SELLER_VKEY=your_seller_verification_key
NETWORK=Preprod  # or Mainnet

# Payment Amounts (Optional)
PAYMENT_AMOUNT=10000000  # 10 ADA in lovelace
PAYMENT_UNIT=lovelace
```

---

## 📖 Usage

### Starting the Agents

#### Explain Agent
```bash
python run_explain_agent.py
```
- Runs on: http://127.0.0.1:8000
- API Docs: http://127.0.0.1:8000/docs

#### Appointment Agent
```bash
python run_appointment_agent.py
```
- Runs on: http://127.0.0.1:8001
- API Docs: http://127.0.0.1:8001/docs

### Running Both Agents Simultaneously

Open two terminal windows:

**Terminal 1:**
```bash
python run_explain_agent.py
```

**Terminal 2:**
```bash
python run_appointment_agent.py
```

---

## 🔌 API Endpoints

### Explain Agent Endpoints

#### Core Endpoints
- `POST /start_job` - Start medical record analysis job
- `POST /upload_pdf` - Upload PDF and get analysis
- `GET /status?job_id={id}` - Get job status
- `GET /health` - Health check

#### Validation Endpoints (New)
- `POST /validate_analysis` - Doctor validates AI analysis
  ```json
  {
    "job_id": "uuid",
    "doctor_id": "DR-001",
    "decision": "approve|reject|modify",
    "comments": "Optional comments"
  }
  ```
- `GET /pending_validations` - Get all pending validations
- `GET /validation_status/{job_id}` - Get validation status

#### Example Request
```bash
curl -X POST "http://127.0.0.1:8000/start_job" \
  -H "Content-Type: application/json" \
  -d '{
    "identifierFromPurchaser": "user-001",
    "input_data": {
      "patient_id": "PATIENT-001",
      "record_text": "Patient: 52 years, BP: 148/92, Glucose: 118 mg/dL, HbA1c: 6.2%",
      "metadata": {
        "timestamp": 1732896000,
        "source": "clinic"
      }
    }
  }'
```

### Appointment Agent Endpoints

#### Core Endpoints
- `POST /start_job` - Start appointment scheduling job
- `GET /status?job_id={id}` - Get job status
- `GET /availability` - Check agent availability
- `GET /input_schema` - Get input schema
- `GET /health` - Health check

#### Example Request
```bash
curl -X POST "http://127.0.0.1:8001/start_job" \
  -H "Content-Type: application/json" \
  -d '{
    "identifierFromPurchaser": "user-001",
    "input_data": {
      "user_request": "I need to see a cardiologist for chest pain",
      "pincode": "110001",
      "patient_info": {
        "name": "John Doe",
        "age": 45,
        "location": "New Delhi"
      }
    }
  }'
```

---

## 🧪 Testing

### Testing Explain Agent

#### Test 1: Normal Values (No Validation)
```json
{
  "patient_id": "TEST-001",
  "record_text": "Patient: 30 years, BP: 120/80, Glucose: 92 mg/dL",
  "metadata": {"timestamp": 1732896000, "source": "test"}
}
```
**Expected:** Low risk, no validation required

#### Test 2: High Risk (Validation Required)
```json
{
  "patient_id": "TEST-002",
  "record_text": "Patient: 52 years, BP: 148/92, Glucose: 118 mg/dL, HbA1c: 6.2%, Family history: diabetes",
  "metadata": {"timestamp": 1732896000, "source": "test"}
}
```
**Expected:** High risk, validation required

#### Test 3: Doctor Validation
```bash
# Check pending validations
curl http://127.0.0.1:8000/pending_validations

# Submit validation
curl -X POST "http://127.0.0.1:8000/validate_analysis" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "job_id=YOUR_JOB_ID&doctor_id=DR-001&decision=approve&comments=Verified"
```

### Testing Appointment Agent

#### Test 1: Cardiology Appointment
```json
{
  "user_request": "I need to see a cardiologist for chest pain",
  "pincode": "110001",
  "patient_info": {"name": "Test", "age": 45, "location": "Delhi"}
}
```

#### Test 2: Gynecology Appointment
```json
{
  "user_request": "I need to see a gynecologist for irregular periods",
  "pincode": "400001",
  "patient_info": {"name": "Test", "age": 32, "location": "Mumbai"}
}
```

See [Testing Guide](./docs/TESTING_GUIDE.md) for comprehensive test scenarios.

---

## 🏗️ Project Structure

```
crewai-masumi-quickstart-template/
├── agents/
│   ├── explain/
│   │   ├── main.py                    # FastAPI app for Explain Agent
│   │   ├── crew_definition.py         # CrewAI agent definitions
│   │   ├── disease_predictor.py        # Disease risk prediction
│   │   ├── guardrails.py              # Input/output safety
│   │   ├── medical_rules.py           # Medical reference ranges
│   │   ├── validation_handler.py      # Doctor validation workflow
│   │   ├── pdf_extractor.py           # PDF text extraction
│   │   └── utils.py                   # Utility functions
│   │
│   └── appointment/
│       ├── appointment_main.py        # FastAPI app for Appointment Agent
│       ├── appointment_crew_definition.py  # CrewAI orchestration
│       ├── hospital_search.py         # Serper API integration
│       ├── hospital_booking.py        # Appointment scheduling
│       └── appointment_utils.py       # Utility functions
│
├── shared/
│   ├── logging_config.py              # Logging configuration
│   └── decision_log.py               # Decision logging
│
├── run_explain_agent.py              # Script to run Explain Agent
├── run_appointment_agent.py          # Script to run Appointment Agent
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 🔐 Security Features

### Explain Agent
- **Input Guardrails**: Blocks code injection, prompt injection, spam
- **Output Guardrails**: Validates medical appropriateness
- **Rule-based Validation**: Prevents hallucinations
- **Doctor Validation**: Human oversight for critical findings

### Appointment Agent
- **Input Validation**: Schema validation for all inputs
- **Error Handling**: Graceful error handling and logging
- **Secure API**: FastAPI with CORS configuration

---

## 🔗 Masumi Network Integration

### Registration

To register your agents on the Masumi Network:

1. **Set up Masumi Node** - Install and run Masumi Payment Service
2. **Fund Wallets** - Add ADA to your selling wallet
3. **Register Agents** - Use Masumi Payment Service API:

```bash
POST /registry/
{
  "network": "Preprod",
  "sellingWalletVkey": "your_vkey",
  "tags": ["medical", "healthcare"],
  "name": "Medical Record Analysis Agent",
  "apiBaseUrl": "http://your-domain.com:8000",
  "description": "AI-powered medical record analysis",
  "capability": {
    "name": "Azure OpenAI GPT-4",
    "version": "1.0.0"
  },
  "agentPricing": {
    "pricingType": "Fixed",
    "pricing": [{"unit": "lovelace", "amount": "10000000"}]
  }
}
```

See [Masumi Documentation](https://docs.masumi.network/documentation) for details.

### Payment Flow

1. Client requests service → Agent creates payment request
2. Client pays via Masumi → Payment confirmed on-chain
3. Agent processes request → Returns results
4. Results logged on-chain → Immutable audit trail

---

## 🛠️ Development

### Adding New Features

1. **Explain Agent**: Add new analysis capabilities in `crew_definition.py`
2. **Appointment Agent**: Extend hospital search in `hospital_search.py`

### Code Style

- Follow PEP 8 Python style guide
- Use type hints where possible
- Add docstrings to all functions
- Log important operations

### Testing

Run tests before committing:
```bash
# Test Explain Agent
python -m pytest tests/test_explain_agent.py

# Test Appointment Agent
python -m pytest tests/test_appointment_agent.py
```

---

## 📊 Features Comparison

| Feature | Explain Agent | Appointment Agent |
|---------|--------------|-------------------|
| Medical Analysis | ✅ | ❌ |
| Disease Prediction | ✅ | ❌ |
| Doctor Validation | ✅ | ❌ |
| Hospital Search | ❌ | ✅ |
| Appointment Scheduling | ❌ | ✅ |
| Specialty Detection | ❌ | ✅ |
| Masumi Integration | ✅ | ✅ |
| MIP-003 Compliance | ✅ | ✅ |

---

## 🐛 Troubleshooting

### Common Issues

1. **Azure OpenAI Connection Error**
   - Verify `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`
   - Check deployment name matches `AZURE_OPENAI_DEPLOYMENT`

2. **Serper API Error (Appointment Agent)**
   - Verify `SERPER_API_KEY` is set
   - Check API quota/limits

3. **Payment Service Connection Error**
   - Ensure Masumi Node is running
   - Check `PAYMENT_SERVICE_URL` is correct
   - Agents will work without payment (fallback mode)

4. **Unicode Encoding Errors (Windows)**
   - Fixed in latest version
   - If issues persist, set `PYTHONIOENCODING=utf-8`

### Logs

Check logs in `logs/app.log` for detailed error information.

---

## 📝 License

MIT License - see LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check documentation in `/docs` folder
- Review Masumi docs: https://docs.masumi.network

---

## 🙏 Acknowledgments

- **CrewAI** - Agent orchestration framework
- **Masumi Network** - Decentralized payment and identity
- **Azure OpenAI** - LLM capabilities
- **Serper** - Real-time search API

---

## 🔄 Version History

### v1.0.0 (Current)
- ✅ Explain Agent with disease prediction
- ✅ Appointment Agent with hospital search
- ✅ Doctor-in-the-loop validation
- ✅ Anti-hallucination protection
- ✅ Masumi Network integration
- ✅ MIP-003 compliance

---

**Built with ❤️ for the MedLedger AI platform**

