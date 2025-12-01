import json
import os
import pandas as pd
import argparse
import sys
import time
from datetime import datetime, timedelta
from models import TextToSQL, SQLOutput
from utils import describe_database

def run_batch_evaluation(db_name, output_file=None, model_name="openai/gpt-4.1-nano"):
    """
    Run all questions for a given database and save results to CSV
    
    Args:
        db_name (str): Name of the database to evaluate
        output_file (str): Optional output CSV filename
        model_name (str): Model name to use for evaluation
    """
    
    # Load the development questions
    print(f"Loading questions for database: {db_name}")
    dev_questions = json.load(open('../../dev.json'))
    dev_questions = pd.DataFrame(dev_questions)
    
    # Filter questions for the specified database
    db_questions = dev_questions[dev_questions['db_id'] == db_name]
    
    if len(db_questions) == 0:
        print(f"❌ No questions found for database: {db_name}")
        available_dbs = sorted(dev_questions['db_id'].unique())
        print(f"Available databases: {', '.join(available_dbs)}")
        return
    
    print(f"Found {len(db_questions)} questions for database: {db_name}")
    
    # Initialize results list
    results = []
    
    # ETA tracking variables
    start_time = time.time()
    question_times = []
    
    # Get database description once (since it's the same for all questions)
    print("Getting database description...")
    sample_t = TextToSQL(
        question="dummy",
        evidence="dummy", 
        expected_sql="dummy",
        difficulty="easy",
        db_name=db_name
    )
    db_description = describe_database(sample_t.path_to_db())
    
    # Process each question
    for question_num, (idx, row) in enumerate(db_questions.iterrows(), 1):
        question_start_time = time.time()
        
        # Calculate ETA after first few questions
        eta_info = ""
        if question_num > 3:  # Wait for a few samples to get reliable average
            avg_time_per_question = sum(question_times) / len(question_times)
            remaining_questions = len(db_questions) - question_num + 1
            estimated_remaining_seconds = avg_time_per_question * remaining_questions
            eta = datetime.now() + timedelta(seconds=estimated_remaining_seconds)
            
            elapsed_total = time.time() - start_time
            eta_info = f" | ETA: {eta.strftime('%H:%M:%S')} (~{estimated_remaining_seconds/60:.1f}m left)"
        
        print(f"\n🔄 Processing question {question_num}/{len(db_questions)}{eta_info}")
        print(f"Question: {row['question'][:100]}...")
        
        try:
            # Create TextToSQL instance
            t = TextToSQL(
                question=row['question'],
                evidence=row['evidence'] if pd.notna(row['evidence']) else "",
                expected_sql=row['SQL'],
                difficulty=row['difficulty'],
                db_name=db_name,
            )
            
            # Make LLM call
            response = t._llm_call(
                prompt=[
                    {
                        'role': 'system', 
                        'content': f'You write sql queries given this database\n\n{db_description}\n\nGive all responses in the JSON format:\n\n{json.dumps(SQLOutput.model_json_schema(), indent=2)}'
                    },
                    {
                        'role': 'user', 
                        'content': f"{t.question}\n\n{t.evidence}"
                    }
                ],
                model_name=model_name,
                openai_api_key=os.getenv("OPENROUTER_API_KEY", "your_openrouter_api_key_here"),
                base_url="https://openrouter.ai/api/v1",
                temperature=0.0
            )
            
            # Initialize result record
            result = {
                'question_id': question_num,  # Use the sequential question number
                'original_idx': idx,  # Keep the original index for reference
                'db_name': db_name,
                'question': row['question'],
                'evidence': row['evidence'] if pd.notna(row['evidence']) else "",
                'expected_sql': row['SQL'],
                'difficulty': row['difficulty'],
                'generated_sql': None,
                'reasoning': None,
                'query_successful': False,
                'results_match': False,
                'generated_result': None,
                'expected_result': None,
                'error_message': None,
                'model_name': model_name,
            }
            
            # Extract generated SQL and reasoning
            if 'parsed' in response and response['parsed']:
                result['generated_sql'] = response['parsed'].sql_query
                result['reasoning'] = response['parsed'].reasoning
                
                # Try to run the generated query
                try:
                    generated_result = t._run_sql_against_db(response['parsed'].sql_query)
                    result['generated_result'] = str(generated_result)
                    result['query_successful'] = True
                    
                    # Compare with expected result
                    expected_result = t.expected_answer()
                    result['expected_result'] = str(expected_result)
                    result['results_match'] = generated_result == expected_result
                    
                    print(f"✅ Query executed successfully. Match: {result['results_match']}")
                    
                except Exception as e:
                    result['error_message'] = str(e)
                    result['query_successful'] = False
                    print(f"❌ Query execution failed: {e}")
            else:
                result['error_message'] = "Failed to parse LLM response"
                print("❌ Failed to parse LLM response")
            
            # Get metadata
            metadata = t.get_metadata()
            result.update({
                'input_tokens': metadata.get('input_tokens', 0),
                'output_tokens': metadata.get('output_tokens', 0),
                'total_tokens': metadata.get('total_tokens', 0),
                'cached_tokens': metadata.get('cached_tokens', 0),
                'latency_seconds': metadata.get('latency_seconds', 0),
                'actual_cost': metadata.get('actual_cost', 0),
                'timestamp': datetime.now().isoformat(),
            })
            
            results.append(result)
            
            # Update ETA tracking
            question_times.append(time.time() - question_start_time)
            
        except Exception as e:
            print(f"❌ Error processing question {question_num}: {e}")
            # Add error record
            error_result = {
                'question_id': question_num,
                'original_idx': idx,
                'db_name': db_name,
                'question': row['question'],
                'evidence': row['evidence'] if pd.notna(row['evidence']) else "",
                'expected_sql': row['SQL'],
                'difficulty': row['difficulty'],
                'generated_sql': None,
                'reasoning': None,
                'query_successful': False,
                'results_match': False,
                'generated_result': None,
                'expected_result': None,
                'error_message': str(e),
                'model_name': model_name,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'cached_tokens': 0,
                'latency_seconds': 0,
                'actual_cost': 0,
                'timestamp': datetime.now().isoformat(),
            }
            results.append(error_result)
            
            # Update ETA tracking even for errors
            question_times.append(time.time() - question_start_time)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clean model name for filename (replace slashes and special chars)
        clean_model_name = model_name.replace("/", "_").replace(":", "_")
        output_file = f"batch_evaluation_{db_name}_{clean_model_name}_{timestamp}.csv"
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    
    # Calculate total execution time
    total_time = time.time() - start_time
    avg_time_per_question = total_time / len(results_df) if len(results_df) > 0 else 0
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"   Total questions: {len(results_df)}")
    print(f"   Successful queries: {results_df['query_successful'].sum()}")
    print(f"   Correct results: {results_df['results_match'].sum()}")
    print(f"   Success rate: {results_df['query_successful'].mean():.2%}")
    print(f"   Accuracy rate: {results_df['results_match'].mean():.2%}")
    
    print(f"\n⏱️  Timing Summary:")
    print(f"   Total execution time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"   Average time per question: {avg_time_per_question:.2f} seconds")
    
    if results_df['actual_cost'].sum() > 0:
        print(f"\n💰 Cost Summary:")
        print(f"   Total cost: ${results_df['actual_cost'].sum():.6f}")
    
    if results_df['latency_seconds'].sum() > 0:
        print(f"   Average API latency: {results_df['latency_seconds'].mean():.3f} seconds")
        print(f"   Total API latency: {results_df['latency_seconds'].sum():.3f} seconds")
    
    print(f"   Total tokens: {results_df['total_tokens'].sum()}")
    
    # Breakdown by difficulty
    if 'difficulty' in results_df.columns:
        print(f"\n📈 Accuracy by Difficulty:")
        difficulty_stats = results_df.groupby('difficulty').agg({
            'results_match': ['count', 'sum', 'mean']
        }).round(3)
        difficulty_stats.columns = ['Total', 'Correct', 'Accuracy']
        print(difficulty_stats)

def main():
    parser = argparse.ArgumentParser(
        description="Run batch evaluation on all questions for a given database"
    )
    parser.add_argument(
        "db_name",
        nargs='?',  # Make db_name optional
        help="Name of the database to evaluate"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output CSV filename (default: auto-generated with timestamp)"
    )
    parser.add_argument(
        "-m", "--model",
        default="openai/gpt-4.1-nano",
        help="Model name to use for evaluation (default: openai/gpt-4.1-nano)"
    )
    parser.add_argument(
        "--list-dbs",
        action="store_true",
        help="List available databases and exit"
    )
    
    args = parser.parse_args()
    
    # Handle list databases option
    if args.list_dbs:
        print("Loading available databases...")
        dev_questions = json.load(open('../dev.json'))
        dev_questions = pd.DataFrame(dev_questions)
        available_dbs = sorted(dev_questions['db_id'].unique())
        print(f"\nAvailable databases ({len(available_dbs)}):")
        for db in available_dbs:
            count = len(dev_questions[dev_questions['db_id'] == db])
            print(f"  - {db} ({count} questions)")
        return
    
    # Check if db_name is provided when not listing databases
    if not args.db_name:
        parser.error("db_name is required when not using --list-dbs")
    
    # Run batch evaluation
    run_batch_evaluation(args.db_name, args.output, args.model)

if __name__ == "__main__":
    main() 