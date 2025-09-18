#!/usr/bin/env python3
"""
Mediimate - WhatsApp Medicine Reminder System (Enhanced)
A complete medicine reminder system using WhatsApp, Gemini AI, and Twilio
Enhanced with smart prescription/report detection and improved database management

New Features:
1. Smart document type detection (prescription vs report)
2. Separate collections for prescriptions and reports
3. Better timing display (AB -> After Breakfast)
4. Complete CRUD operations for medications
5. Enhanced user flows for editing and updating

Usage:
1. Install requirements: pip install fastapi uvicorn twilio google-generativeai python-dotenv python-multipart pyngrok nest_asyncio pymongo pillow requests
2. Set up environment variables in .env file
3. Run: python medicine_reminder_bot_enhanced.py
"""

import os
import io
import json
import time
import random
import datetime
import threading
from threading import Timer
import traceback
import re
from typing import Dict, List, Optional
import requests

# Core imports
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import Response
from twilio.rest import Client
from dotenv import load_dotenv
from bson import ObjectId
from PIL import Image
import uvicorn
import nest_asyncio
import pymongo
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID") 
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = "test"  # Using test database as specified
PRESCRIPTIONS_COLLECTION = "prescriptions"  # New collection for prescriptions
REPORTS_COLLECTION = "reports"  # New collection for reports
USERS_COLLECTION = "users"  # Existing collection for users

# Initialize services
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize MongoDB
try:
    mongo_client = pymongo.MongoClient(MONGODB_URI)
    db = mongo_client[DATABASE_NAME]
    
    # Initialize collections
    prescriptions_collection = db[PRESCRIPTIONS_COLLECTION]
    reports_collection = db[REPORTS_COLLECTION]
    users_collection = db[USERS_COLLECTION]
    
    # Create indexes for better performance
    prescriptions_collection.create_index("user_phone")
    prescriptions_collection.create_index("created_at")
    reports_collection.create_index("user_phone")
    reports_collection.create_index("created_at")
    
    print("📊 Connected to MongoDB successfully!")
    print(f"📋 Using database: {DATABASE_NAME}")
    print(f"💊 Prescriptions collection: {PRESCRIPTIONS_COLLECTION}")
    print(f"📄 Reports collection: {REPORTS_COLLECTION}")
    
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    mongo_client = None
    db = None
    prescriptions_collection = None
    reports_collection = None
    users_collection = None

# FastAPI app
app = FastAPI(title="Mediimate", description="WhatsApp Medicine Reminder System with Smart Document Processing")
nest_asyncio.apply()

# Global storage
user_prescriptions = {}
user_reminders = {}
reminder_threads = {}
user_editing_sessions = {}  # Track users currently editing medications

# Enhanced timing mappings for better readability
TIMING_MAPPINGS = {
    "BB": {"display": "Before Breakfast", "time": "07:30", "food": "before food"},
    "AB": {"display": "After Breakfast", "time": "08:30", "food": "after food"},
    "BL": {"display": "Before Lunch", "time": "12:30", "food": "before food"},
    "AL": {"display": "After Lunch", "time": "13:30", "food": "after food"},
    "BD": {"display": "Before Dinner", "time": "19:30", "food": "before food"},
    "AD": {"display": "After Dinner", "time": "20:30", "food": "after food"},
    "BF": {"display": "Before Food", "time": "12:00", "food": "before food"},
    "AF": {"display": "After Food", "time": "13:00", "food": "after food"},
    "HS": {"display": "At Bedtime", "time": "22:00", "food": "any time"},
    "MORNING": {"display": "Morning", "time": "08:00", "food": "any time"},
    "EVENING": {"display": "Evening", "time": "18:00", "food": "any time"}
}

# Daily health tips/facts
health_tips = [
    "💧 Drink plenty of water to stay hydrated!",
    "🚶‍♀️ A brisk walk every day keeps your heart healthy.",
    "🥗 Eat a variety of fruits and vegetables for balanced nutrition.",
    "🧘‍♂️ Take deep breaths to reduce stress and improve focus.",
    "😴 Getting enough sleep is essential for good health.",
    "🧼 Wash your hands regularly to prevent illness.",
    "🤸‍♀️ Stretch your body to improve flexibility and circulation.",
    "🍭 Limit your sugar intake for better energy and mood.",
    "🙏 Practice gratitude for a positive mindset.",
    "👩‍⚕️ Regular checkups help catch health issues early."
]

# Normal ranges and advice for common lab tests
lab_normal_ranges = {
    "Hemoglobin": {"min": 13.5, "max": 17.5, "unit": "g/dL", "advice": "Eat iron-rich foods like spinach, lentils, and red meat."},
    "WBC": {"min": 4000, "max": 11000, "unit": "/uL", "advice": "Maintain good hygiene and eat immune-boosting foods like citrus fruits."},
    "Platelets": {"min": 150000, "max": 450000, "unit": "/uL", "advice": "Eat papaya, pomegranate, and avoid alcohol."},
    "Blood Sugar (Fasting)": {"min": 70, "max": 99, "unit": "mg/dL", "advice": "Limit sugar intake, exercise regularly, and eat whole grains."},
    "Blood Sugar (PP)": {"min": 70, "max": 140, "unit": "mg/dL", "advice": "Monitor carb intake and stay active after meals."},
    "Cholesterol": {"min": 0, "max": 200, "unit": "mg/dL", "advice": "Reduce saturated fat, eat more fiber, and exercise."},
    "Triglycerides": {"min": 0, "max": 150, "unit": "mg/dL", "advice": "Limit sugary foods, avoid alcohol, and exercise."},
    "Vitamin D": {"min": 20, "max": 50, "unit": "ng/mL", "advice": "Get sunlight exposure and eat fortified foods."},
    "Calcium": {"min": 8.6, "max": 10.2, "unit": "mg/dL", "advice": "Consume dairy, leafy greens, and nuts."},
    "Creatinine": {"min": 0.7, "max": 1.3, "unit": "mg/dL", "advice": "Stay hydrated and avoid excessive protein intake."}
}

def detect_document_type(image_bytes: bytes) -> str:
    """
    Detect whether the uploaded document is a prescription or a medical report
    Returns: 'prescription' or 'report'
    """
    try:
        print(f"🔍 Detecting document type...")
        
        detection_prompt = """
        Analyze this medical document image and determine if it is:
        
        1. PRESCRIPTION: Contains medications, dosage instructions, doctor's signature, pharmacy details, medicine names with timing instructions
        2. REPORT: Contains lab test results, diagnostic values, reference ranges, test parameters, blood work, imaging results
        
        Look for these key indicators:
        
        PRESCRIPTION indicators:
        - Medicine names (e.g., Paracetamol, Aspirin, Metformin)
        - Dosage instructions (e.g., 1-0-1, twice daily, before/after food)
        - Doctor's signature or stamp
        - Pharmacy letterhead
        - "Take as directed" or similar instructions
        - Duration (e.g., "for 7 days")
        
        REPORT indicators:
        - Lab test names (e.g., Hemoglobin, WBC count, Blood Sugar)
        - Numerical values with units (e.g., 12.5 g/dL, 150 mg/dL)
        - Reference ranges (e.g., Normal: 12-16)
        - Hospital/lab letterhead
        - "Results" or "Report" in header
        - Patient demographics and test dates
        
        Respond with only ONE word: "prescription" or "report"
        """
        
        image = Image.open(io.BytesIO(image_bytes))
        response = gemini_model.generate_content([detection_prompt, image])
        
        detection_result = response.text.strip().lower()
        
        # Clean up the response
        if "prescription" in detection_result:
            return "prescription"
        elif "report" in detection_result:
            return "report"
        else:
            # Default to prescription if unclear
            print(f"⚠️ Unclear detection result: {detection_result}, defaulting to prescription")
            return "prescription"
            
    except Exception as e:
        print(f"❌ Document detection failed: {e}")
        # Default to prescription on error
        return "prescription"

def parse_report(image_bytes: bytes) -> tuple:
    """Parse lab report image, flag abnormal values, and suggest improvements."""
    try:
        print(f"🔍 Starting report analysis...")
        print(f"📷 Image size: {len(image_bytes)} bytes")
        
        prompt = """
        You are a medical report parser. Analyze this lab report image and extract a JSON array of test results.
        For each test, return:
        [
          {
            "test": "Test Name",
            "value": "Numeric Value",
            "unit": "Unit",
            "reference_range": "Normal Range if available"
          }
        ]
        Only return the JSON array.
        """
        
        image = Image.open(io.BytesIO(image_bytes))
        response = gemini_model.generate_content([prompt, image])
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3]
        elif response_text.startswith("```"):
            response_text = response_text[3:-3]
            
        print(f"📝 Got response: {response_text[:200]}...")
        
        test_results = json.loads(response_text)
        flagged = []
        
        for result in test_results:
            test = result.get("test")
            value = float(result.get("value", 0))
            unit = result.get("unit", "")
            
            normal = lab_normal_ranges.get(test)
            if normal:
                if value < normal["min"]:
                    flagged.append({
                        "test": test,
                        "value": value,
                        "unit": unit,
                        "flag": "LOW",
                        "advice": normal["advice"]
                    })
                elif value > normal["max"]:
                    flagged.append({
                        "test": test,
                        "value": value,
                        "unit": unit,
                        "flag": "HIGH",
                        "advice": normal["advice"]
                    })
        
        summary = "📊 *Report Analysis*\n\n"
        if flagged:
            for f in flagged:
                summary += f"❗ {f['test']}: {f['value']} {f['unit']} ({f['flag']})\n💡 Advice: {f['advice']}\n\n"
        else:
            summary += "✅ All values are within normal range!\nKeep up the healthy habits."
            
        print(f"✅ Report summary: {summary}")
        return summary, test_results
        
    except Exception as e:
        print(f"❌ Report parsing failed: {e}")
        return "❌ Could not analyze the report. Please try again with a clear image.", []

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text content from PDF bytes"""
    try:
        print(f"📄 Extracting text from PDF...")
        
        # Try using PyPDF2 first
        try:
            import PyPDF2
            import io
            
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            extracted_text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted_text += page.extract_text() + "\n"
                
            if extracted_text.strip():
                print(f"✅ Extracted {len(extracted_text)} characters using PyPDF2")
                return extracted_text.strip()
                
        except ImportError:
            print("⚠️ PyPDF2 not available, using Gemini AI fallback")
        except Exception as e:
            print(f"⚠️ PyPDF2 extraction failed: {e}, using Gemini AI fallback")
        
        # Fallback to Gemini AI for text extraction
        prompt = """
        This is a PDF document. Extract all the text content from this document.
        Return the complete text exactly as it appears, maintaining formatting and structure.
        Include all medical test names, values, units, reference ranges, and patient information.
        Do not add any interpretation, just extract the raw text.
        """
        
        # Note: Gemini can handle PDF directly in some cases
        response = gemini_model.generate_content([prompt, pdf_bytes])
        extracted_text = response.text.strip()
        
        if extracted_text:
            print(f"✅ Extracted {len(extracted_text)} characters using Gemini AI")
            return extracted_text
        else:
            raise Exception("No text extracted from PDF")
        
    except Exception as e:
        print(f"❌ PDF text extraction failed: {e}")
        return ""

async def handle_pdf_document(user_phone: str, pdf_bytes: bytes, media_url: str) -> str:
    """Handle PDF document processing and storage - detects if prescription or medical report"""
    try:
        print(f"📄 Processing PDF document for {user_phone}")
        
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(pdf_bytes)
        
        if not extracted_text:
            return "❌ Could not extract text from PDF. Please try uploading as an image."
        
        # First, determine if this is a prescription or medical report
        document_type_prompt = f"""
        Analyze this document text and determine if it's a PRESCRIPTION or MEDICAL_REPORT.
        
        Text: {extracted_text[:2000]}
        
        PRESCRIPTION indicators:
        - Lists medications/medicines with dosages
        - Contains terms like "mg", "ml", "tablets", "capsules"
        - Has frequency instructions (twice daily, BD, TDS, etc.)
        - Doctor's prescription format
        - Medicine names like Metformin, Paracetamol, etc.
        
        MEDICAL_REPORT indicators:
        - Laboratory test results with values
        - Blood test parameters (glucose, cholesterol, hemoglobin, etc.)
        - Test ranges and reference values
        - Pathology lab report format
        - Contains units like mg/dL, g/dL, etc.
        
        Respond with only: PRESCRIPTION or MEDICAL_REPORT
        """
        
        type_response = gemini_model.generate_content(document_type_prompt)
        document_type = type_response.text.strip().upper()
        
        print(f"🔍 Document type detected: {document_type}")
        
        if "PRESCRIPTION" in document_type:
            # Handle as prescription
            return await handle_pdf_prescription(user_phone, extracted_text)
        else:
            # Handle as medical report (existing logic)
            return await handle_pdf_medical_report(user_phone, extracted_text)
            
    except Exception as e:
        print(f"❌ Error processing PDF: {str(e)}")
        return f"❌ Failed to process PDF: {str(e)}"

async def handle_pdf_prescription(user_phone: str, extracted_text: str) -> str:
    """Handle PDF prescription processing and storage"""
    try:
        print(f"💊 Processing prescription PDF for {user_phone}")
        
        # Use Gemini AI to extract medicines from prescription text
        prescription_prompt = f"""
        You are a prescription parser. Analyze this prescription text and extract medication details.
        
        Text: {extracted_text}
        
        Return a JSON array of medications:
        [
          {{
            "medicine": "Medicine Name",
            "dosage": "Strength (e.g., 500mg)",
            "frequency": "Timing code (e.g., BB, AB, BL, AL, BD, AD)",
            "duration": "Duration (e.g., 7 days)",
            "instructions": "Special instructions if any"
          }}
        ]
        
        Common timing codes:
        - BB: Before Breakfast, AB: After Breakfast  
        - BL: Before Lunch, AL: After Lunch
        - BD: Before Dinner, AD: After Dinner
        - TDS: Three times daily, BD: Twice daily
        
        Only return the JSON array.
        """
        
        ai_response = gemini_model.generate_content(prescription_prompt)
        response_text = ai_response.text.strip()
        
        # Clean up response
        if response_text.startswith("```json"):
            response_text = response_text[7:-3]
        elif response_text.startswith("```"):
            response_text = response_text[3:-3]
        
        print(f"📝 AI prescription response: {response_text[:200]}...")
        
        try:
            medications = json.loads(response_text)
        except json.JSONDecodeError:
            return "❌ Could not parse prescription data. Please try again."
        
        # Process and enhance timing display
        for med in medications:
            timing_code = med.get('frequency', '').upper()
            if timing_code in TIMING_MAPPINGS:
                timing_info = TIMING_MAPPINGS[timing_code]
                med['timing_display'] = timing_info['display']
                med['suggested_time'] = timing_info['time']
                med['food_relation'] = timing_info['food']
            else:
                med['timing_display'] = med.get('frequency', 'As directed')
                med['suggested_time'] = '12:00'
                med['food_relation'] = 'any time'
        
        # Store prescription in database
        prescription_data = {
            "user_phone": user_phone,
            "extracted_medicines": medications,
            "upload_time": datetime.now(),
            "raw_text": extracted_text
        }
        
        result = prescriptions_collection.insert_one(prescription_data)
        prescription_id = str(result.inserted_id)
        
        print(f"💾 Stored prescription with ID: {prescription_id}")
        print(f"📊 Found {len(medications)} medicines")
        
        # Create response message
        summary = "💊 *Prescription Uploaded Successfully!*\n\n"
        summary += f"📋 Found {len(medications)} medicines:\n\n"
        
        for i, med in enumerate(medications, 1):
            summary += f"{i}. *{med['medicine']}* {med.get('dosage', '')}\n"
            summary += f"   ⏰ {med.get('timing_display', 'As directed')}\n"
            if med.get('duration'):
                summary += f"   📅 {med['duration']}\n"
            summary += "\n"
        
        summary += "✅ *Ready for medicine reminders!*\n"
        summary += "💬 Type 'edit prescription' to modify medicines\n"
        summary += "🔔 Type 'reminder on' to start medicine reminders"
        
        return summary
        
    except Exception as e:
        print(f"❌ Error processing prescription PDF: {str(e)}")
        return f"❌ Failed to process prescription: {str(e)}"

async def handle_pdf_medical_report(user_phone: str, extracted_text: str) -> str:
    """Handle PDF medical report processing and storage"""
    try:
        # Use Gemini AI to analyze the extracted text and create structured data
        analysis_prompt = f"""
        Analyze this medical report text and extract structured data.
        
        Text: {extracted_text[:8000]}  # Limit text to avoid token limits
        
        Provide your response in the following format:

        ANALYSIS_JSON:
        {{
            "summary": "Brief summary of overall health status",
            "keyFindings": [
                {{
                    "parameter": "Test name",
                    "value": "Test value with unit", 
                    "status": "normal/high/low",
                    "description": "Clinical significance"
                }}
            ],
            "recommendations": ["List of recommendations"],
            "followUpActions": ["List of follow-up actions"],
            "riskFactors": ["List of identified risk factors"],
            "overallAssessment": "Overall health assessment",
            "urgencyLevel": "low/medium/high"
        }}

        TEST_RESULTS_JSON:
        {{
            "testResults": [
                {{
                    "parameter": "Test name",
                    "value": "numeric value only",
                    "unit": "unit only",
                    "normalRange": {{
                        "min": "minimum value",
                        "max": "maximum value", 
                        "description": "range description"
                    }},
                    "status": "normal/high/low",
                    "category": "test category"
                }}
            ]
        }}

        Extract ALL test parameters you can find in the document. Be thorough.
        """
        
        ai_response = gemini_model.generate_content(analysis_prompt)
        ai_analysis_text = ai_response.text.strip()
        
        print(f"🔍 AI Response length: {len(ai_analysis_text)} characters")
        print(f"📄 AI Response preview: {ai_analysis_text[:500]}...")
        
        # Parse AI analysis with improved logic
        ai_analysis = {}
        test_results = []
        
        try:
            # Extract analysis JSON
            analysis_match = re.search(r'ANALYSIS_JSON:\s*(\{.*?\})\s*TEST_RESULTS_JSON:', ai_analysis_text, re.DOTALL)
            if analysis_match:
                try:
                    ai_analysis = json.loads(analysis_match.group(1))
                    print(f"✅ Parsed analysis JSON successfully")
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse analysis JSON: {e}")
            
            # Extract test results JSON
            test_results_match = re.search(r'TEST_RESULTS_JSON:\s*(\{.*\})', ai_analysis_text, re.DOTALL)
            if test_results_match:
                try:
                    test_data = json.loads(test_results_match.group(1))
                    test_results = test_data.get("testResults", [])
                    print(f"✅ Parsed {len(test_results)} test results")
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse test results JSON: {e}")
            
            # Fallback: Try to find any valid JSON objects
            if not ai_analysis and not test_results:
                print("🔄 Trying fallback JSON extraction...")
                json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', ai_analysis_text, re.DOTALL)
                
                for json_str in json_objects:
                    try:
                        parsed = json.loads(json_str)
                        if "summary" in parsed and not ai_analysis:
                            ai_analysis = parsed
                            print(f"✅ Found analysis via fallback")
                        elif "testResults" in parsed and not test_results:
                            test_results = parsed["testResults"]
                            print(f"✅ Found {len(test_results)} test results via fallback")
                    except:
                        continue
                        
        except Exception as e:
            print(f"❌ JSON parsing error: {e}")
        
        # Analyze test results for red flags and generate personalized recommendations
        flags_analysis = analyze_test_results_with_flags(test_results)
        personalized_recommendations = get_personalized_recommendations(test_results, flags_analysis)
        print(f"🎯 Generated {len(personalized_recommendations)} personalized recommendations")
        
        # Ensure we have at least basic data
        if not ai_analysis:
            ai_analysis = {
                "summary": "PDF processed successfully - comprehensive health analysis completed",
                "keyFindings": [],
                "recommendations": personalized_recommendations,
                "followUpActions": ["Schedule follow-up with healthcare provider if needed"],
                "riskFactors": [],
                "overallAssessment": "Health report analyzed with personalized recommendations",
                "urgencyLevel": "low" if not any("high" in str(result.get("status", "")).lower() for result in test_results) else "medium"
            }
        else:
            # Merge personalized recommendations with AI recommendations
            if ai_analysis.get("recommendations"):
                # Combine and deduplicate recommendations
                all_recommendations = list(ai_analysis["recommendations"])
                for rec in personalized_recommendations:
                    if rec not in all_recommendations:
                        all_recommendations.append(rec)
                ai_analysis["recommendations"] = all_recommendations[:4]  # Limit total recommendations
            else:
                ai_analysis["recommendations"] = personalized_recommendations
        
        print(f"📊 Final results: Analysis={bool(ai_analysis)}, Tests={len(test_results)}")
        
        # Create document record in the format you specified
        document_record = {
            "userId": ObjectId("507f1f77bcf86cd799439012"),  # You may want to create user records
            "reportType": "Blood Test",  # Extract from document or set default
            "title": extract_patient_name(extracted_text) or "Medical Report",
            "originalFileName": f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "extractedText": extracted_text,
            "aiAnalysis": ai_analysis,
            "fileInfo": {
                "originalName": f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                "size": len(extracted_text),  # Fixed: changed from pdf_bytes
                "mimeType": "application/pdf",
                "uploadedAt": datetime.datetime.now()
            },
            "processingStatus": "analyzed",
            "tags": [],
            "isArchived": False,
            "testResults": test_results,
            "trends": [],
            "createdAt": datetime.datetime.now(),
            "updatedAt": datetime.datetime.now()
        }
        
        # Store in reports collection
        if reports_collection is not None:
            try:
                result = reports_collection.insert_one(document_record)
                print(f"📊 Stored PDF report with ID: {result.inserted_id}")
                print(f"🔍 Document stored with {len(test_results)} test results")
                
                # Debug: Print first few test results
                if test_results:
                    print(f"📋 Sample test results:")
                    for i, test in enumerate(test_results[:3]):
                        print(f"  {i+1}. {test.get('parameter', 'N/A')}: {test.get('value', 'N/A')} {test.get('unit', '')}")
                else:
                    print("⚠️ No test results were extracted!")
                
                # Create user-friendly response
                response = f"📊 *PDF Report Analysis Complete*\n\n"
                response += f"📄 *Document:* {document_record['title']}\n"
                response += f"📅 *Processed:* {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                
                if ai_analysis.get("summary"):
                    response += f"📋 *Summary:* {ai_analysis['summary'][:300]}...\n\n"
                
                if ai_analysis.get("keyFindings"):
                    response += f"🔍 *Key Findings:*\n"
                    for finding in ai_analysis["keyFindings"][:3]:  # Show top 3
                        response += f"• {finding.get('parameter', 'N/A')}: {finding.get('status', 'N/A').upper()}\n"
                    response += "\n"
                
                if ai_analysis.get("recommendations"):
                    response += f"💡 *Recommendations:*\n"
                    for rec in ai_analysis["recommendations"][:2]:  # Show top 2
                        response += f"• {rec}\n"
                    response += "\n"
                
                response += f"✅ *Report saved successfully in database!*\n"
                response += f"📊 Total tests analyzed: {len(test_results)}\n"
                
                if len(test_results) == 0:
                    response += f"⚠️ *Note:* No individual test results extracted. This might be due to document format.\n"
                    response += f"📄 Full text analysis is available in database.\n"
                
                response += f"🏥 Access full report anytime with command 'reports'"
                
                return response
            except Exception as db_error:
                print(f"❌ Database storage error: {db_error}")
                return f"❌ Failed to save report to database: {str(db_error)}"
        else:
            return "❌ Database not available. Could not save report."
            
    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        return f"❌ Failed to process PDF: {str(e)}"

def generate_health_recommendations(test_results: List[Dict]) -> List[str]:
    """Generate personalized health recommendations based on test results and severity"""
    try:
        recommendations = []
        high_priority_issues = []
        medium_priority_issues = []
        
        # Analyze each test result
        for test in test_results:
            parameter = test.get("parameter", "").lower()
            status = test.get("status", "").lower()
            value = test.get("value", "")
            
            if status in ["high", "low"]:
                # High priority health issues
                if "hba1c" in parameter or "glycosylated hemoglobin" in parameter:
                    try:
                        hba1c_value = float(str(value).replace("%", ""))
                        if hba1c_value >= 6.5:
                            high_priority_issues.append("diabetes")
                            recommendations.append("🩺 *Diabetes Management:* Switch to millet-based foods (jowar, bajra, ragi), avoid refined sugars, and do 30 minutes of daily exercise")
                        elif hba1c_value >= 5.7:
                            high_priority_issues.append("prediabetes")
                            recommendations.append("⚠️ *Pre-diabetes Alert:* Include millets in your diet, practice portion control, and walk 10,000 steps daily to prevent diabetes")
                    except:
                        pass
                
                elif "cholesterol" in parameter and "total" in parameter:
                    if status == "high":
                        medium_priority_issues.append("high_cholesterol")
                        recommendations.append("❤️ *Heart Health:* Reduce fried foods, include oats and nuts in your diet, and do cardio exercises 3 times a week")
                
                elif "triglycerides" in parameter:
                    if status == "high":
                        medium_priority_issues.append("high_triglycerides")
                        recommendations.append("🥗 *Lipid Control:* Limit sugar and alcohol, eat omega-3 rich foods (fish, walnuts), and maintain healthy weight")
                
                elif "blood pressure" in parameter or "bp" in parameter:
                    if status == "high":
                        high_priority_issues.append("hypertension")
                        recommendations.append("🧂 *Blood Pressure:* Reduce salt intake, practice meditation, and include potassium-rich foods (bananas, spinach)")
                
                elif "vitamin d" in parameter:
                    if status == "low":
                        medium_priority_issues.append("vitamin_d_deficiency")
                        recommendations.append("☀️ *Vitamin D Boost:* Get 15-20 minutes morning sunlight daily, include fortified foods, and consider supplementation")
                
                elif "hemoglobin" in parameter or "hb" in parameter:
                    if status == "low":
                        medium_priority_issues.append("anemia")
                        recommendations.append("🩸 *Iron Deficiency:* Eat iron-rich foods (spinach, dates, jaggery), combine with vitamin C foods, avoid tea with meals")
                
                elif "uric acid" in parameter:
                    if status == "high":
                        medium_priority_issues.append("high_uric_acid")
                        recommendations.append("🦶 *Uric Acid Control:* Drink plenty of water, limit red meat and alcohol, include cherries and low-fat dairy")
                
                elif "creatinine" in parameter:
                    if status == "high":
                        high_priority_issues.append("kidney_function")
                        recommendations.append("🥤 *Kidney Health:* Increase water intake, reduce protein and salt, monitor blood pressure regularly")
                
                elif "thyroid" in parameter or "tsh" in parameter:
                    if status == "high":
                        medium_priority_issues.append("thyroid_issues")
                        recommendations.append("🦋 *Thyroid Care:* Include iodine-rich foods (sea vegetables), practice stress management, ensure regular sleep")
        
        # Add general wellness recommendations if no specific issues found
        if not recommendations:
            recommendations.append("✅ *Good Health Maintenance:* Continue regular exercise, balanced diet with seasonal fruits and vegetables")
            recommendations.append("🏃‍♂️ *Preventive Care:* Annual health checkups, stay hydrated, and maintain work-life balance")
        
        # Limit to 2-3 most important recommendations based on severity
        prioritized_recommendations = []
        
        # Add high priority first (max 2)
        high_priority_recs = [rec for rec in recommendations if any(issue in rec.lower() for issue in ["diabetes", "blood pressure", "kidney"])]
        prioritized_recommendations.extend(high_priority_recs[:2])
        
        # Add medium priority if space available
        medium_priority_recs = [rec for rec in recommendations if rec not in prioritized_recommendations]
        remaining_slots = 3 - len(prioritized_recommendations)
        prioritized_recommendations.extend(medium_priority_recs[:remaining_slots])
        
        # Ensure we have at least 2 recommendations
        if len(prioritized_recommendations) < 2:
            prioritized_recommendations.extend(recommendations[:3])
        
        # Remove duplicates while preserving order
        final_recommendations = []
        for rec in prioritized_recommendations:
            if rec not in final_recommendations:
                final_recommendations.append(rec)
        
        return final_recommendations[:3]  # Maximum 3 recommendations
        
    except Exception as e:
        print(f"❌ Error generating recommendations: {e}")
        return ["🏥 Please consult with your healthcare provider for personalized advice"]

def extract_patient_name(text: str) -> str:
    """Extract patient name from medical report text"""
    try:
        # Look for common patterns
        import re
        patterns = [
            r"Name\s*:\s*([A-Za-z\s\.]+)",
            r"Patient\s*:\s*([A-Za-z\s\.]+)", 
            r"Mr\.\s*([A-Za-z\s]+)",
            r"Mrs\.\s*([A-Za-z\s]+)",
            r"Ms\.\s*([A-Za-z\s]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 2:
                    return name
                    
        return "Unknown Patient"
        
    except Exception as e:
        print(f"❌ Name extraction failed: {e}")
        return "Unknown Patient"

def analyze_test_results_with_flags(test_results: List[Dict]) -> Dict:
    """Analyze test results and identify red flags with severity levels"""
    red_flags = []
    yellow_flags = []
    normal_results = []
    
    try:
        # Handle case where test_results might be a string
        if isinstance(test_results, str):
            print(f"⚠️ Warning: test_results is a string, not a list: {test_results[:100]}...")
            return {
                'red_flags': [],
                'yellow_flags': [],
                'normal_results': [],
                'summary': 'Invalid data format - expected list of test results'
            }
        
        if not isinstance(test_results, list):
            print(f"⚠️ Warning: test_results is not a list: {type(test_results)}")
            return {
                'red_flags': [],
                'yellow_flags': [],
                'normal_results': [],
                'summary': f'Invalid data format - expected list, got {type(test_results)}'
            }
        
        for test in test_results:
            if not isinstance(test, dict):
                print(f"⚠️ Warning: test item is not a dict: {type(test)} - {test}")
                continue
                
            parameter = test.get('parameter', '')
            value_str = test.get('value', '')
            unit = test.get('unit', '')
            status = test.get('status', '').lower()
            normal_range = test.get('normalRange', {})
            
            # Try to convert value to float for range checking
            try:
                value = float(value_str)
            except:
                value = None
            
            # Determine severity based on parameter type and deviation
            severity = "normal"
            deviation_percent = 0
            
            # Special handling for critical parameters
            parameter_lower = parameter.lower()
            
            # Critical parameters that need immediate attention
            if any(keyword in parameter_lower for keyword in ['hba1c', 'hemoglobin a1c']):
                if value and value > 7.0:
                    severity = "high"
                    deviation_percent = ((value - 7.0) / 7.0) * 100
                elif value and value > 6.5:
                    severity = "medium"
                    deviation_percent = ((value - 6.5) / 6.5) * 100
            
            elif any(keyword in parameter_lower for keyword in ['glucose', 'blood sugar']):
                if value and value > 200:
                    severity = "high"
                    deviation_percent = ((value - 140) / 140) * 100
                elif value and value > 140:
                    severity = "medium"
                    deviation_percent = ((value - 100) / 100) * 100
            
            elif any(keyword in parameter_lower for keyword in ['cholesterol', 'ldl']):
                if value and value > 240:
                    severity = "high"
                    deviation_percent = ((value - 200) / 200) * 100
                elif value and value > 200:
                    severity = "medium"
                    deviation_percent = ((value - 200) / 200) * 100
            
            elif any(keyword in parameter_lower for keyword in ['creatinine']):
                if value and value > 1.5:
                    severity = "high"
                    deviation_percent = ((value - 1.2) / 1.2) * 100
                elif value and value > 1.2:
                    severity = "medium"
                    deviation_percent = ((value - 1.0) / 1.0) * 100
            
            elif any(keyword in parameter_lower for keyword in ['hemoglobin', 'hb']):
                if value and value < 10:
                    severity = "high"
                    deviation_percent = ((12 - value) / 12) * 100
                elif value and value < 12:
                    severity = "medium"
                    deviation_percent = ((12 - value) / 12) * 100
            
            elif any(keyword in parameter_lower for keyword in ['vitamin d']):
                if value and value < 20:
                    severity = "medium"
                    deviation_percent = ((30 - value) / 30) * 100
                elif value and value < 30:
                    severity = "low"
                    deviation_percent = ((30 - value) / 30) * 100
            
            # General range-based analysis if no specific handling above
            elif value and normal_range:
                # Handle case where normal_range is a string (like "70-100")
                if isinstance(normal_range, str):
                    # Try to parse string ranges like "70-100" or "<200" or ">40"
                    min_val = None
                    max_val = None
                    try:
                        if '-' in normal_range:
                            parts = normal_range.split('-')
                            if len(parts) == 2:
                                min_val = float(parts[0].strip())
                                max_val = float(parts[1].strip())
                        elif normal_range.startswith('<'):
                            max_val = float(normal_range[1:].strip())
                        elif normal_range.startswith('>'):
                            min_val = float(normal_range[1:].strip())
                    except:
                        pass
                elif isinstance(normal_range, dict):
                    min_val = normal_range.get('min')
                    max_val = normal_range.get('max')
                else:
                    min_val = None
                    max_val = None
                
                if min_val and max_val:
                    try:
                        min_val = float(min_val) if min_val else None
                        max_val = float(max_val) if max_val else None
                        
                        if min_val and max_val:
                            range_size = max_val - min_val
                            
                            if value < min_val:
                                deviation_percent = abs((min_val - value) / min_val) * 100
                            elif value > max_val:
                                deviation_percent = abs((value - max_val) / max_val) * 100
                                
                            # Classify severity based on deviation
                            if deviation_percent > 50:
                                severity = "high"
                            elif deviation_percent > 20:
                                severity = "medium"
                            elif deviation_percent > 0:
                                severity = "low"
                    except:
                        pass
            
            # Also consider status indicators
            if status in ['very high', 'critically high', 'very low', 'critically low']:
                severity = "high"
            elif status in ['high', 'low', 'borderline']:
                if severity == "normal":
                    severity = "medium"
            
            # Create flag entry
            flag_entry = {
                "parameter": parameter,
                "value": f"{value_str} {unit}".strip(),
                "status": status,
                "severity": severity,
                "deviation": round(deviation_percent, 1)
            }
            
            # Categorize flags
            if severity == "high":
                red_flags.append(flag_entry)
            elif severity in ["medium", "low"]:
                yellow_flags.append(flag_entry)
            else:
                normal_results.append(flag_entry)
        
        return {
            "red_flags": red_flags,
            "yellow_flags": yellow_flags,
            "normal_results": normal_results,
            "total_tests": len(test_results)
        }
        
    except Exception as e:
        print(f"❌ Error analyzing test results: {e}")
        return {"red_flags": [], "yellow_flags": [], "normal_results": [], "total_tests": 0}

def get_personalized_recommendations(test_results: List[Dict], flags_analysis: Dict) -> List[str]:
    """Generate personalized health recommendations based on test results and flags"""
    recommendations = []
    
    try:
        # Priority 1: Address red flags first
        for flag in flags_analysis.get('red_flags', []):
            parameter = flag.get('parameter', '').lower()
            value = flag.get('value', '')
            deviation = flag.get('deviation', 0)
            
            # Diabetes/Blood Sugar related
            if any(keyword in parameter for keyword in ['hba1c', 'hemoglobin a1c', 'glycated hemoglobin']):
                recommendations.extend([
                    f"🚨 CRITICAL: HbA1c at {value} indicates poor diabetes control!",
                    "🏥 IMMEDIATE: Schedule urgent appointment with endocrinologist",
                    "🌾 DIET: Replace rice/wheat with millets (bajra, jowar, ragi) - 50% lower glycemic index",
                    "🏃‍♂️ EXERCISE: 30-minute walk after each meal mandatory"
                ])
            elif any(keyword in parameter for keyword in ['glucose', 'sugar', 'blood sugar']):
                try:
                    glucose_val = float(value.split()[0])
                    if glucose_val > 200:
                        recommendations.extend([
                            f"🚨 URGENT: Blood glucose {value} is dangerously high!",
                            "💊 Check medication compliance with doctor immediately",
                            "🚫 AVOID: All sugary foods, fruits, refined carbs for 2 weeks"
                        ])
                    elif glucose_val > 140:
                        recommendations.extend([
                            f"⚠️ High glucose {value} - Take action now!",
                            "🥗 Eat protein + fiber before any carbs",
                            "⏰ Check blood sugar 2 hours after meals"
                        ])
                except:
                    pass
            
            # Cholesterol related
            elif any(keyword in parameter for keyword in ['cholesterol', 'ldl', 'triglycerides']):
                try:
                    chol_val = float(value.split()[0])
                    if chol_val > 240:
                        recommendations.extend([
                            f"🚨 HIGH CHOLESTEROL: {value} - Heart attack risk!",
                            "🏥 URGENT: Cardiologist consultation within 1 week",
                            "🥜 Daily: 6 almonds + 3 walnuts (soaked overnight)",
                            "🐟 Include: Salmon/mackerel 3x/week, AVOID all fried foods"
                        ])
                    else:
                        recommendations.extend([
                            f"⚠️ Elevated cholesterol {value} needs attention",
                            "🥥 Switch to coconut/olive oil for cooking",
                            "🚶‍♂️ Brisk walk 45 minutes daily"
                        ])
                except:
                    pass
            
            # Blood Pressure
            elif any(keyword in parameter for keyword in ['blood pressure', 'bp', 'systolic', 'diastolic']):
                recommendations.extend([
                    f"🚨 HIGH BP: {value} - Stroke/heart attack risk!",
                    "💊 Take BP medication exactly as prescribed",
                    "🧂 STRICT: Salt <1500mg/day (1/4 teaspoon total)",
                    "🧘‍♂️ Practice deep breathing 15 minutes twice daily"
                ])
            
            # Kidney function
            elif any(keyword in parameter for keyword in ['creatinine', 'kidney', 'urea', 'bun']):
                recommendations.extend([
                    f"🚨 KIDNEY CONCERN: {value} needs immediate attention!",
                    "💧 Increase water to 3-4 liters daily",
                    "🥩 REDUCE protein: Limit dal to 1 cup, avoid red meat",
                    "🏥 Nephrology consultation recommended"
                ])
        
        # Priority 2: Address yellow flags with specific actions
        for flag in flags_analysis.get('yellow_flags', []):
            parameter = flag.get('parameter', '').lower()
            value = flag.get('value', '')
            
            # Vitamin deficiencies
            if 'vitamin d' in parameter:
                try:
                    vit_d = float(value.split()[0])
                    if vit_d < 20:
                        recommendations.extend([
                            f"⚠️ Severe Vitamin D deficiency: {value}",
                            "☀️ Sun exposure: 20-30 minutes daily (10am-11am)",
                            "💊 Supplement: D3 60,000 IU weekly (consult doctor)"
                        ])
                    else:
                        recommendations.extend([
                            f"💡 Low Vitamin D: {value} - Easy to fix!",
                            "🥛 Include: Fortified milk, egg yolks, fatty fish"
                        ])
                except:
                    pass
            
            elif 'vitamin b12' in parameter:
                recommendations.extend([
                    f"⚠️ B12 deficiency: {value} affects nerves",
                    "� Include: Lean meat, fish, dairy products",
                    "💊 Consider B12 supplements after blood test"
                ])
            
            elif any(keyword in parameter for keyword in ['hemoglobin', 'hb', 'iron']):
                try:
                    hb_val = float(value.split()[0])
                    if hb_val < 10:
                        recommendations.extend([
                            f"🩸 Severe anemia: {value} - needs treatment",
                            "🥬 Iron-rich: Spinach, beetroot, pomegranate daily",
                            "🍊 With Vitamin C: Lemon water, amla, guava"
                        ])
                    else:
                        recommendations.extend([
                            f"💡 Mild anemia: {value} - dietary changes help",
                            "🥜 Soak: Dates + raisins overnight, eat morning"
                        ])
                except:
                    pass
            
            # Thyroid
            elif any(keyword in parameter for keyword in ['tsh', 'thyroid', 't3', 't4']):
                recommendations.extend([
                    f"⚠️ Thyroid imbalance: {value}",
                    "🧂 Use iodized salt, include seaweed if available",
                    "� Coconut oil beneficial for thyroid function",
                    "🏥 Endocrinology follow-up recommended"
                ])
        
        # Remove duplicates and prioritize based on severity
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        # Add report recommendations based on health conditions detected
        report_recommendations = get_recommended_reports(test_results, flags_analysis)
        
        # If no specific medical recommendations, add general health tips
        if not unique_recommendations:
            unique_recommendations = [
                "✅ Good results! Maintain current healthy lifestyle",
                "🥗 Continue balanced diet with variety of nutrients",
                "🏃‍♂️ Keep up regular exercise routine",
                "🔄 Regular monitoring as recommended by doctor"
            ]
        
        # Always include report recommendations if there are health issues
        if report_recommendations and (flags_analysis.get('red_flags') or flags_analysis.get('yellow_flags')):
            # Limit medical recommendations to make room for report recommendations
            unique_recommendations = unique_recommendations[:5]
            unique_recommendations.extend(report_recommendations)

        # Limit to top 8 most important recommendations for readability
        return unique_recommendations[:8]
        
    except Exception as e:
        print(f"❌ Error generating recommendations: {e}")
        return [
            "🏥 Consult your healthcare provider for detailed guidance",
            "📊 Regular health checkups are important",
            "💊 Take medications as prescribed",
            "🥗 Maintain a balanced diet and regular exercise"
        ]

def get_recommended_reports(test_results: List[Dict], flags_analysis: Dict) -> List[str]:
    """Generate recommendations for follow-up medical reports based on current test results"""
    report_recommendations = []
    
    try:
        # Analyze red flags and yellow flags to recommend specific reports
        all_flags = flags_analysis.get('red_flags', []) + flags_analysis.get('yellow_flags', [])
        
        conditions_detected = set()
        
        for flag in all_flags:
            parameter = flag.get('parameter', '').lower()
            
            # Diabetes/Blood Sugar Issues
            if any(keyword in parameter for keyword in ['hba1c', 'glucose', 'sugar', 'glycated']):
                conditions_detected.add('diabetes')
            
            # Heart/Cardiovascular Issues  
            elif any(keyword in parameter for keyword in ['cholesterol', 'ldl', 'hdl', 'triglycerides']):
                conditions_detected.add('cardiovascular')
            
            # Kidney Issues
            elif any(keyword in parameter for keyword in ['creatinine', 'urea', 'bun', 'kidney']):
                conditions_detected.add('kidney')
            
            # Liver Issues
            elif any(keyword in parameter for keyword in ['alt', 'ast', 'bilirubin', 'liver']):
                conditions_detected.add('liver')
            
            # Anemia/Blood Issues
            elif any(keyword in parameter for keyword in ['hemoglobin', 'hb', 'iron', 'ferritin']):
                conditions_detected.add('anemia')
            
            # Thyroid Issues
            elif any(keyword in parameter for keyword in ['tsh', 'thyroid', 't3', 't4']):
                conditions_detected.add('thyroid')
            
            # Vitamin Deficiencies
            elif any(keyword in parameter for keyword in ['vitamin', 'b12', 'folate']):
                conditions_detected.add('vitamins')
        
        # Generate specific report recommendations based on detected conditions
        if 'diabetes' in conditions_detected:
            report_recommendations.append("📋 *Recommended Reports:* HbA1c monitoring every 3 months, Lipid profile, Kidney function test")
        
        if 'cardiovascular' in conditions_detected:
            report_recommendations.append("❤️ *Heart Health Reports:* ECG, 2D Echo, Stress test, Complete lipid profile every 6 months")
        
        if 'kidney' in conditions_detected:
            report_recommendations.append("🫘 *Kidney Monitoring:* Urine routine, Microalbumin, eGFR, Ultrasound KUB if needed")
        
        if 'liver' in conditions_detected:
            report_recommendations.append("🔬 *Liver Function:* Complete LFT, Hepatitis B & C screening, Ultrasound abdomen")
        
        if 'anemia' in conditions_detected:
            report_recommendations.append("🩸 *Anemia Workup:* Complete Blood Count, Iron studies, B12 & Folate levels, Peripheral smear")
        
        if 'thyroid' in conditions_detected:
            report_recommendations.append("🦋 *Thyroid Monitoring:* TSH, Free T3, Free T4 every 6-8 weeks until normal")
        
        if 'vitamins' in conditions_detected:
            report_recommendations.append("💊 *Vitamin Panel:* Vitamin D, B12, Folate, Iron studies for comprehensive deficiency check")
        
        # If multiple conditions, add comprehensive monitoring
        if len(conditions_detected) >= 2:
            report_recommendations.append("🏥 *Comprehensive Monitoring:* Consider quarterly health checkup with all relevant tests")
        
        # General follow-up recommendations
        if conditions_detected:
            report_recommendations.append("📅 *Follow-up Schedule:* Repeat abnormal tests after 4-6 weeks of treatment/lifestyle changes")
        
        return report_recommendations[:3]  # Limit to top 3 most important reports
        
    except Exception as e:
        print(f"❌ Error generating report recommendations: {e}")
        return []

def store_report_data(user_phone: str, image_bytes: bytes, analysis_result: str, test_results: List[Dict]) -> bool:
    """Store medical report data in MongoDB reports collection"""
    try:
        if reports_collection is None:
            print("❌ Reports collection not available")
            return False
            
        # Create report document
        report_document = {
            "user_phone": user_phone,
            "report_date": datetime.datetime.now(),
            "analysis_summary": analysis_result,
            "test_results": test_results,
            "image_size": len(image_bytes),
            "created_at": datetime.datetime.now(),
            "updated_at": datetime.datetime.now()
        }
        
        # Insert report
        result = reports_collection.insert_one(report_document)
        print(f"📊 Stored report for {user_phone} with ID: {result.inserted_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to store report: {e}")
        return False

def parse_prescription(image_bytes: bytes) -> tuple:
    """Parse prescription image and extract medication details with enhanced timing display."""
    try:
        print(f"💊 Starting prescription parsing...")
        print(f"📷 Image size: {len(image_bytes)} bytes")
        
        prompt = """
        You are a prescription parser. Analyze this prescription image and extract medication details.
        Return a JSON array of medications:
        [
          {
            "medicine": "Medicine Name",
            "dosage": "Strength (e.g., 500mg)",
            "frequency": "Timing code (e.g., BB, AB, BL, AL, BD, AD)",
            "duration": "Duration (e.g., 7 days)",
            "instructions": "Special instructions if any"
          }
        ]
        
        Common timing codes:
        - BB: Before Breakfast
        - AB: After Breakfast  
        - BL: Before Lunch
        - AL: After Lunch
        - BD: Before Dinner
        - AD: After Dinner
        
        Only return the JSON array.
        """
        
        image = Image.open(io.BytesIO(image_bytes))
        response = gemini_model.generate_content([prompt, image])
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3]
        elif response_text.startswith("```"):
            response_text = response_text[3:-3]
            
        print(f"📝 Got response: {response_text[:200]}...")
        
        medications = json.loads(response_text)
        
        # Process and enhance timing display
        for med in medications:
            timing_code = med.get('frequency', '').upper()
            if timing_code in TIMING_MAPPINGS:
                timing_info = TIMING_MAPPINGS[timing_code]
                med['timing_display'] = timing_info['display']
                med['suggested_time'] = timing_info['time']
                med['food_relation'] = timing_info['food']
            else:
                med['timing_display'] = med.get('frequency', 'As directed')
                med['suggested_time'] = '12:00'
                med['food_relation'] = 'any time'
        
        # Create summary with enhanced display
        summary = "💊 *Prescription Analysis*\n\n"
        for i, med in enumerate(medications, 1):
            summary += f"{i}. *{med['medicine']}* {med.get('dosage', '')}\n"
            summary += f"   ⏰ {med['timing_display']}\n"
            summary += f"   📅 Duration: {med.get('duration', 'As prescribed')}\n"
            if med.get('instructions'):
                summary += f"   📝 Instructions: {med['instructions']}\n"
            summary += "\n"
            
        print(f"✅ Prescription summary: {summary}")
        return summary, medications
        
    except Exception as e:
        print(f"❌ Prescription parsing failed: {e}")
        return "❌ Could not parse the prescription. Please try again with a clear image.", []

def store_prescription_data(user_phone: str, image_bytes: bytes, summary: str, medications: List[Dict]) -> bool:
    """Store prescription data in MongoDB prescriptions collection"""
    try:
        if prescriptions_collection is None:
            print("❌ Prescriptions collection not available")
            return False
            
        # Create prescription document
        prescription_document = {
            "user_phone": user_phone,
            "prescription_date": datetime.datetime.now(),
            "summary": summary,
            "medications": medications,
            "image_size": len(image_bytes),
            "status": "active",  # active, completed, cancelled
            "created_at": datetime.datetime.now(),
            "updated_at": datetime.datetime.now()
        }
        
        # Insert prescription
        result = prescriptions_collection.insert_one(prescription_document)
        prescription_id = str(result.inserted_id)
        
        print(f"💊 Stored prescription for {user_phone} with ID: {prescription_id}")
        
        # Also store in memory for quick access
        if user_phone not in user_prescriptions:
            user_prescriptions[user_phone] = []
        user_prescriptions[user_phone].append({
            "id": prescription_id,
            "medications": medications,
            "created_at": datetime.datetime.now()
        })
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to store prescription: {e}")
        return False

def get_user_prescriptions(user_phone: str) -> List[Dict]:
    """Get all prescriptions for a user from database"""
    try:
        if prescriptions_collection is None:
            return []
            
        prescriptions = list(prescriptions_collection.find(
            {"user_phone": user_phone, "status": "active"},
            {"_id": 1, "medications": 1, "created_at": 1, "prescription_date": 1}
        ).sort("created_at", -1))
        
        return prescriptions
        
    except Exception as e:
        print(f"❌ Failed to get prescriptions: {e}")
        return []

def update_medication(user_phone: str, prescription_id: str, medication_index: int, updated_medication: Dict) -> bool:
    """Update a specific medication in a prescription"""
    try:
        if prescriptions_collection is None:
            return False
            
        # Get the prescription
        prescription = prescriptions_collection.find_one({"_id": ObjectId(prescription_id), "user_phone": user_phone})
        if not prescription:
            return False
            
        # Update the specific medication
        medications = prescription.get("medications", [])
        if 0 <= medication_index < len(medications):
            medications[medication_index].update(updated_medication)
            
            # Update in database
            prescriptions_collection.update_one(
                {"_id": ObjectId(prescription_id)},
                {
                    "$set": {
                        "medications": medications,
                        "updated_at": datetime.datetime.now()
                    }
                }
            )
            
            print(f"✅ Updated medication {medication_index} in prescription {prescription_id}")
            return True
            
        return False
        
    except Exception as e:
        print(f"❌ Failed to update medication: {e}")
        return False

def delete_medication(user_phone: str, prescription_id: str, medication_index: int) -> bool:
    """Delete a specific medication from a prescription"""
    try:
        if prescriptions_collection is None:
            return False
            
        # Get the prescription
        prescription = prescriptions_collection.find_one({"_id": ObjectId(prescription_id), "user_phone": user_phone})
        if not prescription:
            return False
            
        # Remove the specific medication
        medications = prescription.get("medications", [])
        if 0 <= medication_index < len(medications):
            medications.pop(medication_index)
            
            # Update in database
            prescriptions_collection.update_one(
                {"_id": ObjectId(prescription_id)},
                {
                    "$set": {
                        "medications": medications,
                        "updated_at": datetime.datetime.now()
                    }
                }
            )
            
            print(f"✅ Deleted medication {medication_index} from prescription {prescription_id}")
            return True
            
        return False
        
    except Exception as e:
        print(f"❌ Failed to delete medication: {e}")
        return False

def show_medication_edit_menu(user_phone: str) -> str:
    """Show menu for editing medications"""
    prescriptions = get_user_prescriptions(user_phone)
    
    if not prescriptions:
        return "❌ No active prescriptions found. Upload a prescription first!"
    
    menu = "📝 *Edit Medications Menu*\n\n"
    
    for i, prescription in enumerate(prescriptions):
        menu += f"*Prescription {i+1} (Added: {prescription['created_at'].strftime('%Y-%m-%d')})\n"
        medications = prescription.get('medications', [])
        for j, med in enumerate(medications):
            timing_display = med.get('timing_display', med.get('frequency', 'As directed'))
            menu += f"  {j+1}. {med['medicine']} - {timing_display}\n"
        menu += "\n"
    
    menu += "📱 *Commands:*\n"
    menu += "• Type `edit P1 M2` to edit Prescription 1, Medication 2\n"
    menu += "• Type `delete P1 M2` to delete Prescription 1, Medication 2\n"
    menu += "• Type `view P1` to view all medications in Prescription 1\n"
    menu += "• Type `menu` to return to main menu\n"
    
    return menu

def handle_medication_edit_command(user_phone: str, command: str) -> str:
    """Handle medication editing commands"""
    try:
        parts = command.lower().split()
        
        if len(parts) < 2:
            return "❌ Invalid command format. Use: edit P1 M2 or delete P1 M2"
            
        action = parts[0]
        
        if action == "view" and len(parts) == 2:
            # View prescription command
            prescription_num = int(parts[1][1:]) - 1  # Remove 'P' and convert to 0-based index
            prescriptions = get_user_prescriptions(user_phone)
            
            if 0 <= prescription_num < len(prescriptions):
                prescription = prescriptions[prescription_num]
                medications = prescription.get('medications', [])
                
                response = f"💊 *Prescription {prescription_num + 1} Details*\n\n"
                for i, med in enumerate(medications):
                    timing_display = med.get('timing_display', med.get('frequency', 'As directed'))
                    response += f"{i+1}. *{med['medicine']}* {med.get('dosage', '')}\n"
                    response += f"   ⏰ {timing_display}\n"
                    response += f"   📅 Duration: {med.get('duration', 'As prescribed')}\n"
                    if med.get('instructions'):
                        response += f"   📝 Instructions: {med['instructions']}\n"
                    response += "\n"
                
                return response
            else:
                return "❌ Invalid prescription number."
                
        elif action in ["edit", "delete"] and len(parts) == 3:
            prescription_num = int(parts[1][1:]) - 1  # Remove 'P' and convert to 0-based index
            medication_num = int(parts[2][1:]) - 1    # Remove 'M' and convert to 0-based index
            
            prescriptions = get_user_prescriptions(user_phone)
            
            if 0 <= prescription_num < len(prescriptions):
                prescription = prescriptions[prescription_num]
                prescription_id = str(prescription['_id'])
                medications = prescription.get('medications', [])
                
                if 0 <= medication_num < len(medications):
                    if action == "delete":
                        if delete_medication(user_phone, prescription_id, medication_num):
                            return f"✅ Deleted medication {medication_num + 1} from prescription {prescription_num + 1}"
                        else:
                            return "❌ Failed to delete medication"
                            
                    elif action == "edit":
                        # Start editing session
                        user_editing_sessions[user_phone] = {
                            "prescription_id": prescription_id,
                            "medication_index": medication_num,
                            "current_medication": medications[medication_num].copy(),
                            "step": "choose_field"
                        }
                        
                        med = medications[medication_num]
                        timing_display = med.get('timing_display', med.get('frequency', 'As directed'))
                        
                        response = f"✏️ *Editing: {med['medicine']}*\n\n"
                        response += f"Current details:\n"
                        response += f"💊 Medicine: {med['medicine']}\n"
                        response += f"💉 Dosage: {med.get('dosage', 'Not specified')}\n"
                        response += f"⏰ Timing: {timing_display}\n"
                        response += f"📅 Duration: {med.get('duration', 'Not specified')}\n"
                        response += f"📝 Instructions: {med.get('instructions', 'None')}\n\n"
                        response += "What would you like to edit?\n"
                        response += "1. Medicine name\n2. Dosage\n3. Timing\n4. Duration\n5. Instructions\n\n"
                        response += "Type the number (1-5) or 'cancel' to abort."
                        
                        return response
                else:
                    return "❌ Invalid medication number."
            else:
                return "❌ Invalid prescription number."
        else:
            return "❌ Invalid command format. Use: edit P1 M2 or delete P1 M2"
            
    except (ValueError, IndexError) as e:
        return "❌ Invalid command format. Use: edit P1 M2 or delete P1 M2"
    except Exception as e:
        print(f"❌ Error handling edit command: {e}")
        return "❌ Error processing command. Please try again."

def handle_editing_session(user_phone: str, message: str) -> str:
    """Handle ongoing medication editing session"""
    try:
        session = user_editing_sessions.get(user_phone)
        if not session:
            return "❌ No active editing session."
            
        if message.lower() == 'cancel':
            del user_editing_sessions[user_phone]
            return "❌ Editing cancelled."
            
        step = session.get("step")
        
        if step == "choose_field":
            field_map = {
                "1": ("medicine", "Enter new medicine name:"),
                "2": ("dosage", "Enter new dosage (e.g., 500mg):"),
                "3": ("timing", "Enter new timing (BB/AB/BL/AL/BD/AD):"),
                "4": ("duration", "Enter new duration (e.g., 7 days):"),
                "5": ("instructions", "Enter new instructions:")
            }
            
            if message in field_map:
                field, prompt = field_map[message]
                session["editing_field"] = field
                session["step"] = "enter_value"
                user_editing_sessions[user_phone] = session
                return f"✏️ {prompt}"
            else:
                return "❌ Invalid choice. Please enter 1-5 or 'cancel'."
                
        elif step == "enter_value":
            field = session["editing_field"]
            
            # Update the medication
            updated_medication = session["current_medication"].copy()
            
            if field == "timing":
                # Handle timing with display conversion
                timing_code = message.upper()
                if timing_code in TIMING_MAPPINGS:
                    timing_info = TIMING_MAPPINGS[timing_code]
                    updated_medication['frequency'] = timing_code
                    updated_medication['timing_display'] = timing_info['display']
                    updated_medication['suggested_time'] = timing_info['time']
                    updated_medication['food_relation'] = timing_info['food']
                else:
                    updated_medication['frequency'] = message
                    updated_medication['timing_display'] = message
            else:
                updated_medication[field] = message
            
            # Save to database
            if update_medication(
                user_phone, 
                session["prescription_id"], 
                session["medication_index"], 
                updated_medication
            ):
                del user_editing_sessions[user_phone]
                timing_display = updated_medication.get('timing_display', updated_medication.get('frequency', 'As directed'))
                
                response = f"✅ *Medication Updated Successfully!*\n\n"
                response += f"💊 Medicine: {updated_medication['medicine']}\n"
                response += f"💉 Dosage: {updated_medication.get('dosage', 'Not specified')}\n"
                response += f"⏰ Timing: {timing_display}\n"
                response += f"📅 Duration: {updated_medication.get('duration', 'Not specified')}\n"
                response += f"📝 Instructions: {updated_medication.get('instructions', 'None')}\n"
                
                return response
            else:
                del user_editing_sessions[user_phone]
                return "❌ Failed to update medication. Please try again."
                
    except Exception as e:
        print(f"❌ Error in editing session: {e}")
        if user_phone in user_editing_sessions:
            del user_editing_sessions[user_phone]
        return "❌ Error during editing. Session cancelled."

def send_whatsapp_message(to_number: str, message: str):
    """Send WhatsApp message via Twilio."""
    try:
        # Ensure both numbers are in the correct format
        from_number = TWILIO_FROM_NUMBER if TWILIO_FROM_NUMBER.startswith("whatsapp:") else f"whatsapp:{TWILIO_FROM_NUMBER}"
        to_number = to_number if to_number.startswith("whatsapp:") else f"whatsapp:{to_number}"

        msg = twilio_client.messages.create(
            body=message,
            from_=from_number,   # always your Twilio number
            to=to_number         # always the user’s number
        )
        print(f"✅ Message sent to {to_number}: {msg.sid}")
    except Exception as e:
        print(f"❌ Failed to send message to {to_number}: {e}")


def setup_reminders(user_phone: str, medications: List[Dict]):
    """Set up medication reminders for a user."""
    try:
        print(f"⏰ Setting up reminders for {user_phone}")
        
        # Stop existing reminders
        if user_phone in reminder_threads:
            print(f"🛑 Stopping existing reminders for {user_phone}")
            for thread in reminder_threads[user_phone]:
                thread.do_run = False
            del reminder_threads[user_phone]
        
        # Store user medications
        user_reminders[user_phone] = medications
        reminder_threads[user_phone] = []
        
        # Start reminder threads for each medication
        for med in medications:
            timing = med.get('suggested_time', '12:00')
            thread = threading.Thread(
                target=reminder_worker,
                args=(user_phone, med, timing)
            )
            thread.do_run = True
            thread.start()
            reminder_threads[user_phone].append(thread)
            
        print(f"✅ Set up {len(medications)} reminders for {user_phone}")
        
    except Exception as e:
        print(f"❌ Failed to setup reminders: {e}")

def reminder_worker(user_phone: str, medication: Dict, timing: str):
    """Worker thread for sending medication reminders."""
    try:
        thread = threading.current_thread()
        print(f"⏰ Starting reminder worker for {medication['medicine']} at {timing}")
        
        while getattr(thread, "do_run", True):
            now = datetime.datetime.now()
            target_time = datetime.datetime.strptime(timing, "%H:%M").time()
            target_datetime = datetime.datetime.combine(now.date(), target_time)
            
            # If target time has passed today, schedule for tomorrow
            if now.time() > target_time:
                target_datetime += datetime.timedelta(days=1)
            
            # Calculate sleep time
            sleep_seconds = (target_datetime - now).total_seconds()
            
            if sleep_seconds > 0:
                print(f"💤 Sleeping {sleep_seconds} seconds until next reminder for {medication['medicine']}")
                time.sleep(min(sleep_seconds, 3600))  # Check every hour max
                
                if not getattr(thread, "do_run", True):
                    break
                    
                # Send reminder if it's time
                current_time = datetime.datetime.now().time()
                if abs((current_time.hour * 60 + current_time.minute) - 
                       (target_time.hour * 60 + target_time.minute)) <= 1:
                    
                    timing_display = medication.get('timing_display', medication.get('frequency', 'As directed'))
                    food_relation = medication.get('food_relation', 'any time')
                    
                    reminder_message = f"💊 *Medicine Reminder*\n\n"
                    reminder_message += f"🔔 Time to take: *{medication['medicine']}*\n"
                    reminder_message += f"💉 Dosage: {medication.get('dosage', 'As prescribed')}\n"
                    reminder_message += f"⏰ Timing: {timing_display}\n"
                    reminder_message += f"🍽️ Take {food_relation}\n\n"
                    reminder_message += f"💡 Health tip: {random.choice(health_tips)}\n\n"
                    reminder_message += "Reply with 'taken' to confirm or 'skip' to skip this dose."
                    
                    send_whatsapp_message(user_phone, reminder_message)
                    
                    # Sleep until next day
                    time.sleep(24 * 3600)
                    
    except Exception as e:
        print(f"❌ Reminder worker error: {e}")

# Enhanced Prescription Editing Functions
async def handle_edit_prescription_command(phone_number: str, message_body: str):
    """Handle prescription editing via WhatsApp commands"""
    try:
        message_lower = message_body.lower().strip()
        
        # Handle specific prescription editing FIRST (edit prescription 1, edit prescription 2, etc.)
        if "edit prescription" in message_lower and any(char.isdigit() for char in message_body):
            import re
            number_match = re.search(r'edit prescription (\d+)', message_lower)
            if number_match:
                prescription_num = int(number_match.group(1))
                return await show_prescription_medicines(phone_number, prescription_num)
        
        # Handle general prescription editing (just "edit prescription")
        elif message_lower == "edit prescription":
            # Get user's current prescriptions
            prescriptions = list(prescriptions_collection.find({"user_phone": phone_number}))
            
            if not prescriptions:
                return "❌ No prescriptions found. Please upload a prescription first to edit."
            
            # Show available prescriptions with numbers
            response = "📝 *Edit Prescription*\n\n"
            response += "Your current prescriptions:\n\n"
            
            for i, prescription in enumerate(prescriptions, 1):
                doc_name = prescription.get('document_name', f'Prescription {i}')
                date_created = prescription.get('created_at', 'Unknown date')
                response += f"{i}. {doc_name}\n   📅 {date_created}\n\n"
            
            response += "💬 *How to edit:*\n"
            response += "Reply with: `edit prescription [number]`\n"
            response += "Example: `edit prescription 1`\n\n"
            response += "*Other commands:*\n"
            response += "• `edit medicine [medicine name]` - Edit specific medicine\n"
            response += "• `remove medicine [medicine name]` - Remove medicine\n"
            response += "• `add medicine [name] [dosage] [frequency] [duration]` - Add new medicine"
            
            return response
        
        # Handle specific medicine editing
        elif "edit medicine" in message_lower:
            medicine_name = message_body.lower().replace("edit medicine", "").strip()
            return await show_medicine_edit_options(phone_number, medicine_name)
        
        # Handle medicine property updates
        elif any(cmd in message_lower for cmd in ["update dosage", "update timing", "update frequency", "update duration"]):
            return await process_medicine_update(phone_number, message_body)
        
        # Handle removing medicine
        elif "remove medicine" in message_lower:
            medicine_name = message_body.lower().replace("remove medicine", "").strip()
            return await remove_medicine_from_prescription(phone_number, medicine_name)
        
        # Handle adding new medicine
        elif "add medicine" in message_lower:
            # Format: "add medicine NAME dosage DOSAGE frequency FREQUENCY duration DURATION"
            return await add_medicine_to_prescription(phone_number, message_body)
        
        return None
        
    except Exception as e:
        print(f"❌ Error handling edit prescription: {e}")
        return "❌ Error processing edit request. Please try again."

async def show_prescription_medicines(phone_number: str, prescription_num: int):
    """Show medicines in a specific prescription for editing"""
    try:
        prescriptions = list(prescriptions_collection.find({"user_phone": phone_number}))
        
        if prescription_num > len(prescriptions) or prescription_num < 1:
            return "❌ Invalid prescription number. Please check and try again."
        
        prescription = prescriptions[prescription_num - 1]
        medicines = prescription.get('medications', [])  # Fixed: changed from 'medicines'
        
        if not medicines:
            return "❌ No medicines found in this prescription."
        
        response = f"💊 *Prescription {prescription_num} - Medicines*\n\n"
        
        for i, medicine in enumerate(medicines, 1):
            name = medicine.get('medicine', 'Unknown Medicine')
            dosage = medicine.get('dosage', 'Not specified')
            frequency = medicine.get('frequency', 'Not specified')
            duration = medicine.get('duration', 'Not specified')
            instructions = medicine.get('instructions', 'Not specified')
            
            response += f"{i}. *{name}*\n"
            response += f"   💊 Dosage: {dosage}\n"
            response += f"   ⏰ Frequency: {frequency}\n"
            response += f"   📅 Duration: {duration}\n"
            response += f"   📝 Instructions: {instructions}\n\n"
        
        response += "💬 *Edit Commands:*\n"
        response += "• `edit medicine [medicine name]`\n"
        response += "• `update dosage [medicine] to [new dosage]`\n"
        response += "• `update frequency [medicine] to [new frequency]`\n"
        response += "• `update duration [medicine] to [new duration]`\n"
        response += "• `remove medicine [medicine name]`\n"
        response += "• `add medicine [name] [dosage] [frequency] [duration]`\n\n"
        response += "*Examples:*\n"
        response += "`update dosage ECOSPRIN to 100mg`\n"
        response += "`remove medicine ECOSPRIN`\n"
        response += "`add medicine Vitamin-D 60000IU Weekly 8weeks`"
        
        return response
        
    except Exception as e:
        print(f"❌ Error showing prescription medicines: {e}")
        return "❌ Error retrieving prescription details."

async def show_medicine_edit_options(phone_number: str, medicine_name: str):
    """Show editing options for a specific medicine"""
    try:
        # Find the medicine across all prescriptions
        prescriptions = list(prescriptions_collection.find({"user_phone": phone_number}))
        found_medicine = None
        prescription_id = None
        
        for prescription in prescriptions:
            medicines = prescription.get('medications', [])  # Fixed: changed from 'medicines'
            for medicine in medicines:
                if medicine_name.lower() in medicine.get('medicine', '').lower():  # Fixed: changed from 'name'
                    found_medicine = medicine
                    prescription_id = prescription.get('_id')
                    break
            if found_medicine:
                break
        
        if not found_medicine:
            return f"❌ Medicine '{medicine_name}' not found in your prescriptions.\nTry typing the exact medicine name or use 'edit prescription [number]' to see all medicines."
        
        name = found_medicine.get('medicine', 'Unknown Medicine')  # Fixed: changed from 'name'
        dosage = found_medicine.get('dosage', 'Not specified')
        timing = found_medicine.get('timing', 'Not specified')
        frequency = found_medicine.get('frequency', 'Not specified')
        duration = found_medicine.get('duration', 'Not specified')
        
        response = f"💊 *Editing: {name}*\n\n"
        response += f"Current Details:\n"
        response += f"💊 Dosage: {dosage}\n"
        response += f"⏰ Timing: {timing}\n"
        response += f"📅 Frequency: {frequency}\n"
        response += f"⏳ Duration: {duration}\n\n"
        
        response += "💬 *Update Commands:*\n"
        response += f"• `update dosage {name} to [new dosage]`\n"
        response += f"• `update timing {name} to [new timing]`\n"
        response += f"• `update frequency {name} to [new frequency]`\n"
        response += f"• `update duration {name} to [new duration]`\n\n"
        
        response += "Examples:\n"
        response += f"• `update dosage {name} to 500mg`\n"
        response += f"• `update timing {name} to After Breakfast, After Dinner`"
        
        return response
        
    except Exception as e:
        print(f"❌ Error showing medicine edit options: {e}")
        return "❌ Error retrieving medicine details."

def convert_timing_to_display(timing: str) -> str:
    """Convert timing codes to readable display text"""
    try:
        timing_parts = timing.split(',')
        display_parts = []
        
        for part in timing_parts:
            part = part.strip().upper()
            if part in TIMING_MAPPINGS:
                display_parts.append(TIMING_MAPPINGS[part]['display'])
            else:
                display_parts.append(part)
        
        return ', '.join(display_parts)
    except:
        return timing

async def process_medicine_update(phone_number: str, message_body: str):
    """Process medicine property updates"""
    try:
        import re
        
        # Parse the update command
        update_patterns = {
            'dosage': r'update dosage (.+?) to (.+)',
            'timing': r'update timing (.+?) to (.+)',
            'frequency': r'update frequency (.+?) to (.+)',
            'duration': r'update duration (.+?) to (.+)'
        }
        
        update_type = None
        medicine_name = None
        new_value = None
        
        for update_key, pattern in update_patterns.items():
            match = re.search(pattern, message_body.lower())
            if match:
                update_type = update_key
                medicine_name = match.group(1).strip()
                new_value = match.group(2).strip()
                break
        
        if not all([update_type, medicine_name, new_value]):
            return "❌ Invalid update format. Use: `update [property] [medicine name] to [new value]`"
        
        # Find and update the medicine
        prescriptions = list(prescriptions_collection.find({"user_phone": phone_number}))
        updated = False
        
        for prescription in prescriptions:
            medicines = prescription.get('medications', [])  # Fixed: changed from 'medicines'
            for i, medicine in enumerate(medicines):
                if medicine_name.lower() in medicine.get('medicine', '').lower():  # Fixed: changed from 'name'
                    # Update the specific property
                    medicines[i][update_type] = new_value
                    
                    # If updating timing, also update timing_display
                    if update_type == 'timing':
                        medicines[i]['timing_display'] = convert_timing_to_display(new_value)
                    
                    # Update in database
                    prescriptions_collection.update_one(
                        {"_id": prescription['_id']},
                        {"$set": {"medications": medicines, "updated_at": datetime.now().isoformat()}}  # Fixed: changed to 'medications'
                    )
                    
                    updated = True
                    
                    # Send confirmation
                    response = f"✅ *Updated Successfully!*\n\n"
                    response += f"💊 Medicine: {medicine.get('medicine')}\n"  # Fixed: changed from 'name'
                    response += f"📝 {update_type.title()}: {new_value}\n\n"
                    response += "🔄 Your prescription has been updated in the database."
                    
                    return response
        
        if not updated:
            return f"❌ Medicine '{medicine_name}' not found. Please check the name and try again."
        
    except Exception as e:
        print(f"❌ Error processing medicine update: {e}")
        return "❌ Error updating medicine. Please try again."

async def remove_medicine_from_prescription(phone_number: str, medicine_name: str):
    """Remove a medicine from user's prescription"""
    try:
        if not medicine_name:
            return "❌ Please specify medicine name. Example: `remove medicine ECOSPRIN`"
        
        # Find all prescriptions for the user
        prescriptions = list(prescriptions_collection.find({"user_phone": phone_number}))
        
        if not prescriptions:
            return "❌ No prescriptions found."
        
        # Look for the medicine across all prescriptions
        medicine_found = False
        prescription_updated = None
        
        for prescription in prescriptions:
            medicines = prescription.get('medications', [])
            original_count = len(medicines)
            
            # Remove medicine if found (case-insensitive)
            medicines = [med for med in medicines if medicine_name.lower() not in med.get('medicine', '').lower()]
            
            if len(medicines) < original_count:
                # Medicine was found and removed
                medicine_found = True
                prescription_updated = prescription
                
                # Update the prescription in database
                prescriptions_collection.update_one(
                    {"_id": prescription["_id"]},
                    {"$set": {"medications": medicines, "updated_at": datetime.datetime.now()}}
                )
                
                removed_medicine_name = next(
                    (med.get('medicine', medicine_name.upper()) for med in prescription.get('medications', []) 
                     if medicine_name.lower() in med.get('medicine', '').lower()), 
                    medicine_name.upper()
                )
                
                response = f"✅ *Medicine Removed Successfully!*\n\n"
                response += f"❌ Removed: *{removed_medicine_name}*\n"
                response += f"📊 Remaining medicines: {len(medicines)}\n\n"
                response += "💊 Type `edit prescription` to see updated list"
                
                return response
        
        if not medicine_found:
            return f"❌ Medicine '{medicine_name}' not found in any prescription.\n💡 Type `edit prescription` to see available medicines."
        
    except Exception as e:
        print(f"❌ Error removing medicine: {e}")
        return "❌ Error removing medicine. Please try again."

async def add_medicine_to_prescription(phone_number: str, message_body: str):
    """Add a new medicine to user's prescription"""
    try:
        # Parse the add medicine command
        # Expected format: "add medicine NAME dosage DOSAGE frequency FREQUENCY duration DURATION"
        # Or simplified: "add medicine NAME DOSAGE FREQUENCY DURATION"
        
        parts = message_body.lower().replace("add medicine", "").strip().split()
        
        if len(parts) < 3:
            return ("❌ Invalid format. Use one of these:\n\n"
                   "*Method 1:*\n"
                   "`add medicine [NAME] [DOSAGE] [FREQUENCY] [DURATION]`\n"
                   "Example: `add medicine Paracetamol 500mg BD 7days`\n\n"
                   "*Method 2:*\n"
                   "`add medicine [NAME] dosage [DOSAGE] frequency [FREQUENCY] duration [DURATION]`\n"
                   "Example: `add medicine Paracetamol dosage 500mg frequency BD duration 7days`")
        
        # Try to parse both formats
        medicine_name = ""
        dosage = ""
        frequency = ""
        duration = ""
        instructions = "As directed"
        
        # Check if using keyword format (dosage, frequency, duration keywords)
        if "dosage" in message_body.lower() or "frequency" in message_body.lower():
            # Keyword format parsing
            text = message_body.lower().replace("add medicine", "").strip()
            
            # Extract medicine name (everything before first keyword)
            keywords = ["dosage", "frequency", "duration"]
            first_keyword_pos = min([text.find(kw) for kw in keywords if kw in text] + [len(text)])
            medicine_name = text[:first_keyword_pos].strip()
            
            # Extract other fields
            import re
            dosage_match = re.search(r'dosage\s+([^\s]+(?:\s+[^\s]+)*?)(?:\s+(?:frequency|duration)|$)', text)
            frequency_match = re.search(r'frequency\s+([^\s]+(?:\s+[^\s]+)*?)(?:\s+(?:dosage|duration)|$)', text)
            duration_match = re.search(r'duration\s+([^\s]+(?:\s+[^\s]+)*?)(?:\s+(?:dosage|frequency)|$)', text)
            
            dosage = dosage_match.group(1).strip() if dosage_match else "Not specified"
            frequency = frequency_match.group(1).strip() if frequency_match else "As needed"
            duration = duration_match.group(1).strip() if duration_match else "As prescribed"
            
        else:
            # Simple format parsing
            medicine_name = parts[0]
            dosage = parts[1] if len(parts) > 1 else "Not specified"
            frequency = parts[2] if len(parts) > 2 else "As needed"
            duration = parts[3] if len(parts) > 3 else "As prescribed"
        
        if not medicine_name:
            return "❌ Medicine name is required."
        
        # Create new medicine object
        new_medicine = {
            "medicine": medicine_name.upper(),
            "dosage": dosage,
            "frequency": frequency,
            "duration": duration,
            "instructions": instructions,
            "timing_display": frequency,
            "suggested_time": "12:00",
            "food_relation": "any time"
        }
        
        # Find user's prescriptions and add to the most recent one
        prescriptions = list(prescriptions_collection.find({"user_phone": phone_number}).sort("created_at", -1))
        
        if not prescriptions:
            return "❌ No prescriptions found. Please upload a prescription first."
        
        # Add to the most recent prescription
        latest_prescription = prescriptions[0]
        current_medicines = latest_prescription.get('medications', [])
        current_medicines.append(new_medicine)
        
        # Update the prescription in database
        prescriptions_collection.update_one(
            {"_id": latest_prescription["_id"]},
            {"$set": {"medications": current_medicines, "updated_at": datetime.datetime.now()}}
        )
        
        response = f"✅ *Medicine Added Successfully!*\n\n"
        response += f"💊 *{new_medicine['medicine']}*\n"
        response += f"   💊 Dosage: {new_medicine['dosage']}\n"
        response += f"   ⏰ Frequency: {new_medicine['frequency']}\n"
        response += f"   📅 Duration: {new_medicine['duration']}\n\n"
        response += f"📊 Total medicines: {len(current_medicines)}\n\n"
        response += "💡 Type `edit prescription` to see updated list"
        
        return response
        
    except Exception as e:
        print(f"❌ Error adding medicine: {e}")
        return "❌ Error adding medicine. Please try again."

# Daily Health Tips Scheduler
# Daily health tips database
DAILY_HEALTH_TIPS = [
    "💧 Start your day with 2 glasses of warm water to boost metabolism and hydration",
    "🥗 Eat the rainbow! Include 5 different colored fruits/vegetables in your daily diet",
    "🚶‍♂️ Take a 10-minute walk after each meal to improve digestion and blood sugar",
    "😴 Maintain 7-8 hours of quality sleep for better immunity and mental health",
    "🧘‍♂️ Practice deep breathing for 5 minutes to reduce stress and lower blood pressure",
    "🍃 Include turmeric and ginger in your diet - natural anti-inflammatory powerhouses",
    "📱 Follow the 20-20-20 rule: Every 20 mins, look at something 20 feet away for 20 seconds",
    "🥜 Have a handful of nuts daily - great source of healthy fats and protein",
    "🌞 Get 15-20 minutes of morning sunlight for natural Vitamin D synthesis",
    "🧴 Drink green tea instead of regular tea - packed with antioxidants",
    "🏃‍♂️ Take stairs instead of elevators - simple way to stay active",
    "🥛 Include probiotics like yogurt, buttermilk for better gut health",
    "🍋 Add lemon to your water - helps with digestion and Vitamin C",
    "📖 Read for 30 minutes daily to keep your mind sharp and reduce stress",
    "🧘‍♀️ Practice gratitude - write 3 things you're thankful for each day",
    "🥕 Eat carrots and leafy greens for better eye health and vision",
    "💪 Do 10 push-ups or squats during TV commercial breaks",
    "🥥 Use coconut oil for cooking - healthier alternative to refined oils",
    "🍅 Include tomatoes in your diet - rich in lycopene for heart health",
    "🎵 Listen to music while working - can improve productivity and mood",
    "🍎 Have an apple a day - fiber, vitamins, and natural energy boost",
    "🧊 Apply ice packs for 10 mins on sore muscles for quick relief",
    "🌿 Grow herbs like mint, basil at home for fresh, chemical-free seasoning",
    "🥤 Replace sugary drinks with coconut water or fresh fruit juices",
    "🛁 Take a warm bath with Epsom salt to relax muscles and reduce stress",
    "🥬 Include spinach in your diet - iron, folate, and vitamins galore",
    "🚴‍♂️ Cycle or walk to nearby places instead of driving",
    "🍵 Drink chamomile tea before bed for better sleep quality",
    "🥒 Stay hydrated with cucumber water - refreshing and detoxifying",
    "🤝 Connect with friends and family regularly for better mental health"
]

def schedule_daily_health_tip(phone_number: str, preferred_time: str = "09:00"):
    """Schedule daily health tips for a user"""
    try:
        from datetime import datetime, timedelta
        import random
        
        # Convert preferred time to 24-hour format if needed
        try:
            hour, minute = map(int, preferred_time.split(':'))
        except:
            hour, minute = 9, 0  # Default to 9:00 AM
        
        # Calculate seconds until next scheduled time
        now = datetime.now()
        next_tip_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # If the time has passed today, schedule for tomorrow
        if next_tip_time <= now:
            next_tip_time += timedelta(days=1)
        
        delay = (next_tip_time - now).total_seconds()
        
        # Schedule the first tip
        timer = Timer(delay, send_daily_health_tip, args=[phone_number])
        timer.start()
        
        print(f"✅ Daily health tips scheduled for {phone_number} at {preferred_time}")
        return True
        
    except Exception as e:
        print(f"❌ Error scheduling daily tips: {e}")
        return False

def send_daily_health_tip(phone_number: str):
    """Send a daily health tip and schedule the next one"""
    try:
        import random
        
        # Select a random health tip
        tip = random.choice(DAILY_HEALTH_TIPS)
        
        # Create message
        message = f"🌟 *Daily Health Tip*\n\n{tip}\n\n"
        message += "💚 Stay healthy, stay happy!\n"
        message += "Reply 'stop tips' to unsubscribe from daily tips."
        
        # Send message via Twilio
        try:
            message_sent = twilio_client.messages.create(
                body=message,
                from_='whatsapp:+14155238886',  # Twilio sandbox number
                to=phone_number
            )
            print(f"✅ Daily tip sent to {phone_number}: {message_sent.sid}")
        except Exception as e:
            print(f"❌ Failed to send daily tip to {phone_number}: {e}")
        
        # Schedule next tip for tomorrow at the same time
        timer = Timer(86400, send_daily_health_tip, args=[phone_number])  # 24 hours = 86400 seconds
        timer.start()
        
    except Exception as e:
        print(f"❌ Error sending daily health tip: {e}")

def get_daily_health_tip():
    """Get a random daily health tip"""
    return random.choice(DAILY_HEALTH_TIPS)

async def handle_daily_tips_command(phone_number: str, message_body: str):
    """Handle daily health tips commands"""
    try:
        command = message_body.lower().strip()
        
        if command in ["daily tips", "health tips", "start tips"]:
            response = "🌟 *Daily Health Tips*\n\n"
            response += "Get personalized health tips delivered daily!\n\n"
            response += "⏰ When would you like to receive tips?\n"
            response += "Reply with time in format: HH:MM\n"
            response += "Examples: 09:00, 18:30, 07:45\n\n"
            response += "Default: 09:00 AM if no time specified"
            return response
        
        elif command == "stop tips":
            # In a real implementation, you'd store this preference in database
            # For now, we'll just confirm
            response = "❌ Daily health tips have been stopped.\n\n"
            response += "You can restart anytime by typing 'daily tips'"
            return response
        
        # Check if user provided a time
        import re
        time_pattern = r'^(\d{1,2}):(\d{2})$'
        match = re.match(time_pattern, command)
        
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                preferred_time = f"{hour:02d}:{minute:02d}"
                
                # Schedule daily tips
                if schedule_daily_health_tip(phone_number, preferred_time):
                    response = f"✅ *Daily Tips Activated!*\n\n"
                    response += f"⏰ Time: {preferred_time} daily\n"
                    response += f"📧 You'll receive health tips every day at this time\n\n"
                    response += f"💡 First tip coming up tomorrow!\n"
                    response += f"Reply 'stop tips' anytime to unsubscribe"
                    return response
                else:
                    return "❌ Error setting up daily tips. Please try again."
            else:
                return "❌ Invalid time format. Use HH:MM (e.g., 09:00, 18:30)"
        
        return None
        
    except Exception as e:
        print(f"❌ Error handling daily tips command: {e}")
        return "❌ Error processing daily tips request."

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    """Handle incoming WhatsApp messages."""
    try:
        # Parse incoming form data
        form_data = await request.form()
        print(f"📱 Received webhook data: {dict(form_data)}")

        # Extract message details
        message_body = form_data.get("Body", "").strip()
        from_number = form_data.get("From", "")
        num_media = int(form_data.get("NumMedia", 0))
        media_url = form_data.get("MediaUrl0") if num_media > 0 else None
        media_content_type = form_data.get("MediaContentType0") if num_media > 0 else None

        print(f"📞 From: {from_number}")
        print(f"💬 Message: {message_body}")
        print(f"📎 Media: {media_url} ({media_content_type})")

        response_message = None

        # --- Handle editing sessions ---
        if from_number in user_editing_sessions:
            response_message = handle_editing_session(from_number, message_body)

        # --- Handle media uploads (images / PDFs) ---
        elif media_url and media_content_type and (
            media_content_type.startswith("image/") or media_content_type == "application/pdf"
        ):
            print(f"📄 Processing document from {from_number}: {media_content_type}")

            media_response = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
            if media_response.status_code == 200:
                document_bytes = media_response.content
                if media_content_type == "application/pdf":
                    response_message = await handle_pdf_document(from_number, document_bytes, media_url)
                else:
                    doc_type = detect_document_type(document_bytes)
                    if doc_type == "prescription":
                        summary, medications = parse_prescription(document_bytes)
                        if medications and store_prescription_data(from_number, document_bytes, summary, medications):
                            response_message = (
                                summary
                                + "\n\n✅ Prescription saved!\n"
                                + "💊 Would you like me to set reminders?\n"
                                + "Reply 'yes' to start or 'edit' to modify."
                            )
                        else:
                            response_message = summary
                    elif doc_type == "report":
                        summary, test_results = parse_report(document_bytes)
                        if test_results:
                            flags = analyze_test_results_with_flags(test_results)
                            recs = get_personalized_recommendations(test_results, flags)
                            if store_report_data(from_number, document_bytes, summary, test_results):
                                response_message = summary + "\n\n✅ Report saved!\n"
                                if flags["red_flags"]:
                                    response_message += "🚨 *RED FLAGS:*\n"
                                    for f in flags["red_flags"]:
                                        response_message += f"• {f['parameter']}: {f['value']} (Deviation {f['deviation']}%)\n"
                                if flags["yellow_flags"]:
                                    response_message += "\n⚠️ *ATTENTION REQUIRED:*\n"
                                    for f in flags["yellow_flags"]:
                                        response_message += f"• {f['parameter']}: {f['value']}\n"
                                if recs:
                                    response_message += "\n💡 *RECOMMENDATIONS:*\n"
                                    for r in recs:
                                        response_message += f"• {r}\n"
                            else:
                                response_message = summary
                        else:
                            response_message = summary
            else:
                response_message = "❌ Could not download the document. Please try again."

        # --- Handle text commands ---
        elif message_body:
            msg = message_body.lower().strip()

            if msg in ["hi", "hello", "hey", "start", "help", "namaste"]:
                response_message = (
                    "👋 *Welcome to Mediimate!*\n\n"
                    "Here’s what I can do:\n"
                    "• 📄 Read & analyze prescriptions/reports\n"
                    "• 💊 Extract medicines & set reminders\n"
                    "• 📝 Edit or update medications\n"
                    "• 📊 Analyze lab reports & flag results\n"
                    "• 🌟 Send daily health tips\n"
                    "• 🔔 Remind you to take medicines\n\n"
                    "👉 Send me a prescription/report image or type 'help' to see commands."
                )

            elif msg.startswith("add "):
                response_message = add_medicine_to_prescription(from_number, message_body)

            elif msg in ["show", "list", "medicines"]:
                response_message = show_prescription_medicines(from_number)

            elif msg.startswith("edit "):
                response_message = handle_edit_prescription_command(from_number, message_body)

            elif msg.startswith("remove "):
                response_message = remove_medicine_from_prescription(from_number, message_body)

            elif msg == "yes":
                prescriptions = get_user_prescriptions(from_number)
                if prescriptions:
                    latest = prescriptions[0]
                    meds = latest.get("medications", [])
                    if meds:
                        setup_reminders(from_number, meds)
                        response_message = (
                            f"✅ Set up reminders for {len(meds)} medicines!\n"
                            "💡 Reply 'taken' when you take a dose or 'skip' if you skip."
                        )
                    else:
                        response_message = "❌ No medications found in your latest prescription."
                else:
                    response_message = "❌ No prescriptions found. Please upload one first."

            elif msg == "reminders":
                prescriptions = get_user_prescriptions(from_number)
                if prescriptions and prescriptions[0].get("medications"):
                    meds = prescriptions[0]["medications"]
                    response_message = "⏰ *Your Active Reminders:*\n\n"
                    for i, med in enumerate(meds, 1):
                        timing = med.get("timing_display", med.get("frequency", "As directed"))
                        response_message += f"{i}. {med['medicine']} - {timing}\n"
                else:
                    response_message = "❌ No active reminders. Upload a prescription!"

            elif msg == "stop":
                if from_number in reminder_threads:
                    for thread in reminder_threads[from_number]:
                        thread.do_run = False
                    reminder_threads.pop(from_number, None)
                    user_reminders.pop(from_number, None)
                    response_message = "🛑 All reminders stopped."
                else:
                    response_message = "❌ No active reminders to stop."

            elif msg in ["taken", "skip"]:
                action = "taken" if msg == "taken" else "skipped"
                response_message = f"✅ Dose {action}!\n💡 Health tip: {random.choice(health_tips)}"

            else:
                response_message = (
                    "❓ I didn’t understand that.\n"
                    "Try one of: add, show, edit, remove, reminders, stop, help."
                )

        # --- Send reply ---
        if response_message:
            send_whatsapp_message(from_number, response_message)
            print(f"📤 Sent reply to {from_number}: {response_message}")

        return Response(status_code=200)

    except Exception as e:
        print(f"❌ Webhook error: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return Response(status_code=500)

@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    try:
        form_data = await request.form()
        print(f"📱 Received webhook data: {dict(form_data)}")

        message_body = form_data.get("Body", "").strip()
        from_number = form_data.get("From", "")

        print(f"📞 From: {from_number}")
        print(f"💬 Message: {message_body}")

        response_message = ""

        # --- Your existing logic here ---
        if from_number in user_editing_sessions:
            response_message = handle_editing_session(from_number, message_body)
        else:
            # Handle text commands
            msg = message_body.lower().strip()

            if msg in ["hi", "hello", "hey", "start", "help", "namaste"]:
                response_message = (
                    "👋 *Welcome to Mediimate!*\n\n"
                    "Here’s what I can do:\n"
                    "• 📄 Read & analyze prescriptions/reports\n"
                    "• 💊 Extract medicines & set reminders\n"
                    "• 📝 Edit or update medications\n"
                    "• 📊 Analyze lab reports & flag results\n"
                    "• 🌟 Send daily health tips\n"
                    "• 🔔 Remind you to take medicines\n\n"
                    "👉 Send me a prescription/report image or type 'help' to see commands."
                )

            elif msg.startswith("add "):
                response_message = await add_medicine_to_prescription(from_number, message_body)

            elif msg in ["show", "list", "medicines"]:
                response_message = await show_prescription_medicines(from_number)

            elif msg.startswith("edit "):
                response_message = await handle_edit_prescription_command(from_number, message_body)

            elif msg.startswith("remove "):
                response_message = await remove_medicine_from_prescription(from_number, message_body)

            elif msg == "yes":
                prescriptions = get_user_prescriptions(from_number)
                if prescriptions:
                    latest = prescriptions[0]
                    meds = latest.get("medications", [])
                    if meds:
                        setup_reminders(from_number, meds)
                        response_message = (
                            f"✅ Set up reminders for {len(meds)} medicines!\n"
                            "💡 Reply 'taken' when you take a dose or 'skip' if you skip."
                        )
                    else:
                        response_message = "❌ No medications found in your latest prescription."
                else:
                    response_message = "❌ No prescriptions found. Please upload one first."

            elif msg == "reminders":
                prescriptions = get_user_prescriptions(from_number)
                if prescriptions and prescriptions[0].get("medications"):
                    meds = prescriptions[0]["medications"]
                    response_message = "⏰ *Your Active Reminders:*\n\n"
                    for i, med in enumerate(meds, 1):
                        timing = med.get("timing_display", med.get("frequency", "As directed"))
                        response_message += f"{i}. {med['medicine']} - {timing}\n"
                else:
                    response_message = "❌ No active reminders. Upload a prescription!"
            elif msg == "stop":
                if from_number in reminder_threads:
                    for thread in reminder_threads[from_number]:
                        thread.do_run = False
                    reminder_threads.pop(from_number, None)
                    user_reminders.pop(from_number, None)
                    response_message = "🛑 All reminders stopped."
                else:
                    response_message = "❌ No active reminders to stop."
            elif msg in ["taken", "skip"]:
                action = "taken" if msg == "taken" else "skipped"
                response_message = f"✅ Dose {action}!\n💡 Health tip: {random.choice(health_tips)}"
            else:
                response_message = (
                    "❓ I didn’t understand that.\n"
                    "Try one of: add, show, edit, remove, reminders, stop, help."
                )


        # --- Send reply via Twilio REST API ---
        if response_message:
            twilio_client.messages.create(
                body=response_message,
                from_=TWILIO_FROM_NUMBER,
                to=from_number
            )
            print(f"📤 Sent reply to {from_number}: {response_message}")

        return {"status": "ok"}

    except Exception as e:
        print(f"❌ Webhook error: {e}")
        return {"status": "error", "error": str(e)}


@app.get("/")
def home():
    return {"message": "Hello from Railway!"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "Mediimate",
        "features": [
            "Smart document detection",
            "Enhanced timing display",
            "Complete medication CRUD",
            "Secure database storage"
        ],
        "collections": {
            "prescriptions": PRESCRIPTIONS_COLLECTION,
            "reports": REPORTS_COLLECTION,
            "users": USERS_COLLECTION
        }
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        stats = {
            "active_users": len(user_reminders),
            "active_reminder_threads": sum(len(threads) for threads in reminder_threads.values()),
            "editing_sessions": len(user_editing_sessions)
        }
        
        if prescriptions_collection is not None and reports_collection is not None:
            stats["total_prescriptions"] = prescriptions_collection.count_documents({})
            stats["total_reports"] = reports_collection.count_documents({})
        
        return stats
    except Exception as e:
        print(f"❌ Stats error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    try:
        print("🚀 Starting Mediimate...")
        # Use Railway's dynamic PORT environment variable
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run(app, host="0.0.0.0", port=port)
    except KeyboardInterrupt:
        print("\n🛑 Stopping Mediimate...")
        # Clean up reminder threads
        for user_threads in reminder_threads.values():
            for thread in user_threads:
                thread.do_run = False
        print("✅ Bot stopped successfully!")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
