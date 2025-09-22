#!/usr/bin/env python3
"""
Test script to verify the padding parameter fix
"""
import os
import sys
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline

def test_pipeline_fix():
    print("=== TESTING PIPELINE PARAMETER FIX ===")
    print()
    
    MODEL_NAME = "google/flan-t5-base"
    print(f"Loading model: {MODEL_NAME}")
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("✅ Tokenizer loaded successfully")
        
        print("Loading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        print("✅ Model loaded successfully")
        
        print("Creating pipeline...")
        # Test the fixed pipeline configuration (without padding parameter)
        pipe = pipeline(
            "text2text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=150,
            max_length=None,
            temperature=0.3,
            repetition_penalty=1.2,
            do_sample=True,
            truncation=True  # No padding parameter
        )
        print("✅ Pipeline created successfully (no padding error)")
        
        print("Wrapping in LangChain HuggingFacePipeline...")
        llm = HuggingFacePipeline(pipeline=pipe)
        print("✅ LangChain wrapper created successfully")
        
        print("Testing LLM with a simple query...")
        test_prompt = "What is a lease agreement?"
        response = llm.invoke(test_prompt)
        print(f"✅ LLM response: {response}")
        
        print()
        print("🎉 SUCCESS! The padding parameter fix is working correctly!")
        print("✅ No model_kwargs errors")
        print("✅ Pipeline creates successfully")
        print("✅ LLM responds to queries")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        if "padding" in str(e).lower():
            print("⚠️  The padding parameter is still causing issues")
        else:
            print("⚠️  Different error occurred")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_pipeline_fix()
