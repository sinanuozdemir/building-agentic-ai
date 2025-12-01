import json
import pandas as pd
import argparse
import sys
import time
from datetime import datetime, timedelta
from run_batch_evaluation import run_batch_evaluation

def run_multi_model_evaluation(db_name, models, output_dir=".", max_retries=1):
    """
    Run evaluation on multiple models for a given database
    
    Args:
        db_name (str): Name of the database to evaluate
        models (list): List of model names to evaluate
        output_dir (str): Directory to save results
        max_retries (int): Number of retries for failed models
    """
    
    print(f"🚀 Starting multi-model evaluation on database: {db_name}")
    print(f"📋 Models to evaluate: {len(models)}")
    for i, model in enumerate(models, 1):
        print(f"   {i}. {model}")
    
    # Track results
    results_summary = []
    start_time = time.time()
    
    for model_idx, model_name in enumerate(models, 1):
        model_start_time = time.time()
        
        # Calculate ETA
        if model_idx > 1:
            avg_time_per_model = (time.time() - start_time) / (model_idx - 1)
            remaining_models = len(models) - model_idx + 1
            estimated_remaining = avg_time_per_model * remaining_models
            eta = datetime.now() + timedelta(seconds=estimated_remaining)
            eta_info = f" | ETA: {eta.strftime('%H:%M:%S')} (~{estimated_remaining/60:.1f}m left)"
        else:
            eta_info = ""
        
        print(f"\n{'='*80}")
        print(f"🔄 Model {model_idx}/{len(models)}: {model_name}{eta_info}")
        print(f"{'='*80}")
        
        success = False
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"🔄 Retry attempt {attempt}/{max_retries}")
                
                # Generate output filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                clean_model_name = model_name.replace("/", "_").replace(":", "_").replace("-", "_")
                output_file = f"{output_dir}/batch_evaluation_{db_name}_{clean_model_name}_{timestamp}.csv"
                
                # Run evaluation
                run_batch_evaluation(db_name, output_file, model_name)
                
                # Read the results to get summary statistics
                if output_file:
                    try:
                        df = pd.read_csv(output_file)
                        results_summary.append({
                            'model_name': model_name,
                            'total_questions': len(df),
                            'successful_queries': df['query_successful'].sum(),
                            'correct_results': df['results_match'].sum(),
                            'success_rate': df['query_successful'].mean(),
                            'accuracy_rate': df['results_match'].mean(),
                            'total_cost': df['actual_cost'].sum(),
                            'avg_latency': df['latency_seconds'].mean(),
                            'total_tokens': df['total_tokens'].sum(),
                            'execution_time_minutes': (time.time() - model_start_time) / 60,
                            'output_file': output_file,
                            'status': 'success'
                        })
                        success = True
                        break
                    except Exception as e:
                        print(f"⚠️  Warning: Could not read results file: {e}")
                        success = True  # Still count as success if evaluation ran
                        break
                        
            except Exception as e:
                print(f"❌ Error with model {model_name} (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    results_summary.append({
                        'model_name': model_name,
                        'status': 'failed',
                        'error': str(e),
                        'execution_time_minutes': (time.time() - model_start_time) / 60,
                    })
                else:
                    time.sleep(5)  # Wait before retry
        
        if success:
            print(f"✅ Completed {model_name} in {(time.time() - model_start_time)/60:.1f} minutes")
        else:
            print(f"❌ Failed {model_name} after {max_retries + 1} attempts")
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"🏁 MULTI-MODEL EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"⏱️  Total execution time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"📊 Results summary:")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results_summary)
    
    # Save summary
    summary_file = f"{output_dir}/multi_model_summary_{db_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"💾 Summary saved to: {summary_file}")
    
    # Print summary table
    if len(summary_df) > 0:
        successful_models = summary_df[summary_df['status'] == 'success']
        if len(successful_models) > 0:
            print(f"\n📈 Performance Comparison:")
            comparison_cols = ['model_name', 'accuracy_rate', 'success_rate', 'total_cost', 'avg_latency', 'execution_time_minutes']
            display_df = successful_models[comparison_cols].copy()
            display_df['accuracy_rate'] = display_df['accuracy_rate'].apply(lambda x: f"{x:.1%}")
            display_df['success_rate'] = display_df['success_rate'].apply(lambda x: f"{x:.1%}")
            display_df['total_cost'] = display_df['total_cost'].apply(lambda x: f"${x:.6f}")
            display_df['avg_latency'] = display_df['avg_latency'].apply(lambda x: f"{x:.2f}s")
            display_df['execution_time_minutes'] = display_df['execution_time_minutes'].apply(lambda x: f"{x:.1f}m")
            
            print(display_df.to_string(index=False))
            
            # Best performers
            print(f"\n🏆 Best Performers:")
            best_accuracy = successful_models.loc[successful_models['accuracy_rate'].idxmax()]
            fastest = successful_models.loc[successful_models['execution_time_minutes'].idxmin()]
            cheapest = successful_models.loc[successful_models['total_cost'].idxmin()]
            
            print(f"   🎯 Best Accuracy: {best_accuracy['model_name']} ({best_accuracy['accuracy_rate']:.1%})")
            print(f"   ⚡ Fastest: {fastest['model_name']} ({fastest['execution_time_minutes']:.1f}m)")
            print(f"   💰 Cheapest: {cheapest['model_name']} (${cheapest['total_cost']:.6f})")
        
        failed_models = summary_df[summary_df['status'] == 'failed']
        if len(failed_models) > 0:
            print(f"\n❌ Failed Models ({len(failed_models)}):")
            for _, row in failed_models.iterrows():
                print(f"   - {row['model_name']}: {row.get('error', 'Unknown error')}")
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on multiple models")
    parser.add_argument("db_name", help="Name of the database to evaluate (use 'all' to run on all databases)")
    parser.add_argument("-m", "--models", nargs='+', 
                       help="List of model names to evaluate")
    parser.add_argument("-o", "--output-dir", default=".", 
                       help="Directory to save results (default: current directory)")
    parser.add_argument("--retries", type=int, default=1,
                       help="Number of retries for failed models (default: 1)")
    
    args = parser.parse_args()
    
    # Get list of available databases
    import os
    available_databases = []
    db_dir = "../../dbs/dev_databases"
    if os.path.exists(db_dir):
        available_databases = [d for d in os.listdir(db_dir) 
                             if os.path.isdir(os.path.join(db_dir, d)) and not d.startswith('.')]
    
    # Determine which databases to run on
    if args.db_name.lower() == "all":
        databases_to_run = available_databases
        print(f"🎯 Running evaluation on ALL {len(databases_to_run)} databases:")
        for i, db in enumerate(databases_to_run, 1):
            print(f"   {i}. {db}")
    else:
        databases_to_run = [args.db_name]
        print(f"🎯 Running evaluation on database: {args.db_name}")
       
    # Determine which models to use
    models = [
            'anthropic/claude-sonnet-4',
            'google/gemini-2.5-pro-preview',
            'openai/gpt-4o-mini',
            # 'openai/gpt-4.1-mini', 
            'openai/gpt-4.1-nano',
            'meta-llama/llama-3-8b-instruct',
            'mistralai/mistral-7b-instruct'
        ]
    
    print(f"📋 Using {len(models)} models: {', '.join(models)}")
    
    # Run the evaluation for each database
    all_summaries = []
    total_start_time = time.time()
    
    for db_idx, db_name in enumerate(databases_to_run, 1):
        if len(databases_to_run) > 1:
            print(f"\n{'='*100}")
            print(f"🗄️  DATABASE {db_idx}/{len(databases_to_run)}: {db_name}")
            print(f"{'='*100}")
        
        try:
            summary = run_multi_model_evaluation(
                db_name=db_name,
                models=models,
                output_dir=args.output_dir,
                max_retries=args.retries
            )
            
            # Add database name to summary
            summary['database'] = db_name
            all_summaries.append(summary)
            
        except Exception as e:
            print(f"❌ Failed to run evaluation on database {db_name}: {e}")
            continue
    
    # Final summary for all databases
    if len(databases_to_run) > 1:
        total_time = time.time() - total_start_time
        print(f"\n{'='*100}")
        print(f"🏁 ALL DATABASES EVALUATION COMPLETE")
        print(f"{'='*100}")
        print(f"⏱️  Total execution time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
        print(f"📊 Evaluated {len(databases_to_run)} databases with {len(models)} models each")
        
        # Save combined summary
        if all_summaries:
            import pandas as pd
            combined_summary = pd.concat(all_summaries, ignore_index=True)
            combined_file = f"{args.output_dir}/all_databases_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            combined_summary.to_csv(combined_file, index=False)
            print(f"💾 Combined summary saved to: {combined_file}")
    
    print(f"\n🎉 Multi-model evaluation complete!")

if __name__ == "__main__":
    main() 