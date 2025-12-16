#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation script for the CURZA dataset.
Checks dataset structure, requirements, and quality.
"""

import json
from typing import Dict, List


def validate_dataset(dataset_path: str = "datasets/curza_dataset.json") -> bool:
    """
    Validate the CURZA dataset.
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    print("=" * 80)
    print("VALIDATING CURZA DATASET")
    print("=" * 80)
    
    try:
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"\n✓ Successfully loaded dataset from {dataset_path}")
        
        # Check basic structure
        if not isinstance(dataset, list):
            print("✗ Dataset must be a list")
            return False
        
        if len(dataset) == 0:
            print("✗ Dataset is empty")
            return False
        
        print(f"✓ Dataset contains {len(dataset)} intents")
        
        # Validate each intent
        errors = []
        warnings = []
        stats = {
            'total_intents': len(dataset),
            'total_questions': 0,
            'min_questions': float('inf'),
            'max_questions': 0,
            'intents_with_urls': 0,
            'intents_below_20': 0
        }
        
        for i, intent in enumerate(dataset):
            # Check required fields
            if 'intent' not in intent:
                errors.append(f"Intent {i} missing 'intent' field")
                continue
            
            intent_name = intent['intent']
            
            if 'questions' not in intent:
                errors.append(f"Intent '{intent_name}' missing 'questions' field")
                continue
            
            if 'answer' not in intent:
                errors.append(f"Intent '{intent_name}' missing 'answer' field")
                continue
            
            # Validate questions
            questions = intent['questions']
            if not isinstance(questions, list):
                errors.append(f"Intent '{intent_name}' questions must be a list")
                continue
            
            num_questions = len(questions)
            stats['total_questions'] += num_questions
            stats['min_questions'] = min(stats['min_questions'], num_questions)
            stats['max_questions'] = max(stats['max_questions'], num_questions)
            
            # Check minimum requirement (20 questions)
            if num_questions < 20:
                errors.append(
                    f"Intent '{intent_name}' has only {num_questions} questions "
                    f"(minimum is 20)"
                )
                stats['intents_below_20'] += 1
            
            # Check for duplicate questions
            unique_questions = set(questions)
            if len(unique_questions) != num_questions:
                duplicates = num_questions - len(unique_questions)
                warnings.append(
                    f"Intent '{intent_name}' has {duplicates} duplicate question(s)"
                )
            
            # Check for empty questions
            empty_questions = [q for q in questions if not q or not q.strip()]
            if empty_questions:
                errors.append(
                    f"Intent '{intent_name}' has {len(empty_questions)} empty question(s)"
                )
            
            # Validate answer
            answer = intent['answer']
            if not answer or not answer.strip():
                errors.append(f"Intent '{intent_name}' has empty answer")
            
            # Check if answer contains URL
            if 'http' in answer:
                stats['intents_with_urls'] += 1
            else:
                warnings.append(
                    f"Intent '{intent_name}' answer does not contain a URL"
                )
        
        # Print statistics
        print("\n" + "=" * 80)
        print("STATISTICS")
        print("=" * 80)
        print(f"Total intents: {stats['total_intents']}")
        print(f"Total questions: {stats['total_questions']}")
        print(f"Average questions per intent: {stats['total_questions'] / stats['total_intents']:.1f}")
        print(f"Minimum questions in any intent: {stats['min_questions']}")
        print(f"Maximum questions in any intent: {stats['max_questions']}")
        print(f"Intents with URLs in answers: {stats['intents_with_urls']}/{stats['total_intents']}")
        
        # Print validation results
        print("\n" + "=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)
        
        if errors:
            print(f"\n✗ Found {len(errors)} error(s):")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
        else:
            print("\n✓ No errors found")
        
        if warnings:
            print(f"\n⚠ Found {len(warnings)} warning(s):")
            for warning in warnings[:10]:  # Show first 10 warnings
                print(f"  - {warning}")
            if len(warnings) > 10:
                print(f"  ... and {len(warnings) - 10} more warnings")
        else:
            print("✓ No warnings")
        
        # Check requirements
        print("\n" + "=" * 80)
        print("REQUIREMENTS CHECK")
        print("=" * 80)
        
        requirements_met = True
        
        # Requirement 1: Minimum 20 questions per intent
        if stats['intents_below_20'] == 0:
            print("✓ All intents have at least 20 questions")
        else:
            print(f"✗ {stats['intents_below_20']} intent(s) have less than 20 questions")
            requirements_met = False
        
        # Requirement 2: All answers contain URLs
        if stats['intents_with_urls'] == stats['total_intents']:
            print("✓ All answers contain URLs")
        else:
            missing = stats['total_intents'] - stats['intents_with_urls']
            print(f"⚠ {missing} answer(s) do not contain URLs")
        
        # Requirement 3: No structural errors
        if len(errors) == 0:
            print("✓ Dataset structure is valid")
        else:
            print(f"✗ Found {len(errors)} structural error(s)")
            requirements_met = False
        
        # Final result
        print("\n" + "=" * 80)
        if requirements_met and len(errors) == 0:
            print("✅ VALIDATION PASSED")
            print("=" * 80)
            print("Dataset is ready for use!")
            return True
        else:
            print("❌ VALIDATION FAILED")
            print("=" * 80)
            print("Please fix the errors and run validation again.")
            return False
        
    except FileNotFoundError:
        print(f"✗ Dataset file not found: {dataset_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def main():
    """Main function."""
    import sys
    
    # Get dataset path from command line or use default
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/curza_dataset.json"
    
    # Validate dataset
    success = validate_dataset(dataset_path)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
