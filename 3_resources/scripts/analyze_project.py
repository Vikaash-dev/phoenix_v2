"""
Complete File Analysis - Phoenix Protocol
==========================================

Comprehensive analysis of all files in the project.
Generated: January 6, 2026

Author: Phoenix Protocol Team
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import ast
import json


class ProjectAnalyzer:
    """Analyze entire Phoenix Protocol project."""
    
    def __init__(self, root_dir: str = '.'):
        """Initialize analyzer."""
        self.root_dir = Path(root_dir)
        self.analysis = {
            'python_files': [],
            'documentation': [],
            'total_lines': 0,
            'total_functions': 0,
            'total_classes': 0,
            'features_implemented': {},
            'dependencies': set(),
            'missing_features': []
        }
    
    def analyze_python_file(self, filepath: Path) -> Dict:
        """Analyze single Python file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
            
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # Count non-comment lines
            lines = [l for l in content.split('\n') if l.strip() and not l.strip().startswith('#')]
            
            return {
                'path': str(filepath),
                'lines': len(lines),
                'functions': functions,
                'classes': classes,
                'imports': list(set(imports)),
                'function_count': len(functions),
                'class_count': len(classes)
            }
        except Exception as e:
            return {
                'path': str(filepath),
                'error': str(e),
                'lines': 0,
                'functions': [],
                'classes': [],
                'imports': [],
                'function_count': 0,
                'class_count': 0
            }
    
    def analyze_all_files(self):
        """Analyze all Python files in project."""
        print("=" * 80)
        print("PHOENIX PROTOCOL - COMPLETE FILE ANALYSIS")
        print("=" * 80)
        print()
        
        # Find all Python files
        python_files = list(self.root_dir.glob('**/*.py'))
        python_files = [f for f in python_files if '.git' not in str(f)]
        
        print(f"📁 Found {len(python_files)} Python files")
        print()
        
        # Analyze each file
        for pyfile in sorted(python_files):
            analysis = self.analyze_python_file(pyfile)
            self.analysis['python_files'].append(analysis)
            self.analysis['total_lines'] += analysis['lines']
            self.analysis['total_functions'] += analysis['function_count']
            self.analysis['total_classes'] += analysis['class_count']
            self.analysis['dependencies'].update(analysis['imports'])
        
        # Find documentation files
        doc_files = list(self.root_dir.glob('**/*.md'))
        doc_files = [f for f in doc_files if '.git' not in str(f)]
        
        for doc in sorted(doc_files):
            try:
                size = doc.stat().st_size
                self.analysis['documentation'].append({
                    'path': str(doc),
                    'size_kb': size / 1024
                })
            except:
                pass
        
        print(f"📄 Found {len(doc_files)} documentation files")
        print()
    
    def categorize_files(self) -> Dict[str, List]:
        """Categorize files by purpose."""
        categories = {
            'Core Architecture': [],
            'Data Pipeline': [],
            'Training Infrastructure': [],
            'Deployment': [],
            'Testing': [],
            'P1 Features': [],
            'P2 Features': [],
            'Utilities': [],
            'Documentation': []
        }
        
        for file_info in self.analysis['python_files']:
            path = file_info['path']
            
            if 'models/' in path:
                categories['Core Architecture'].append(file_info)
            elif any(x in path for x in ['data_deduplication', 'physics_informed', 'clinical_preprocessing']):
                categories['Data Pipeline'].append(file_info)
            elif any(x in path for x in ['train', 'phoenix_optimizer', 'training_improvements']):
                categories['Training Infrastructure'].append(file_info)
            elif any(x in path for x in ['onnx', 'quantization', 'clinical_postprocessing']):
                categories['Deployment'].append(file_info)
            elif 'test' in path or 'validate' in path:
                categories['Testing'].append(file_info)
            elif 'p1_features' in path:
                categories['P1 Features'].append(file_info)
            elif 'p2_features' in path:
                categories['P2 Features'].append(file_info)
            else:
                categories['Utilities'].append(file_info)
        
        for doc_info in self.analysis['documentation']:
            categories['Documentation'].append(doc_info)
        
        return categories
    
    def check_feature_completeness(self) -> Dict:
        """Check which features are implemented."""
        features = {
            'P0_Critical': {
                'Mixed Precision Training': False,
                'K-Fold Cross-Validation': False,
                'ONNX Export': False,
                'TFLite Export': False,
                'Reproducible Training': False,
                'Advanced LR Schedulers': False,
                'Gradient Clipping': False,
                'Early Stopping': False,
                'Patient-Level Splitting': False,
                'Model Validation': False,
                'Performance Benchmarking': False
            },
            'P1_Important': {
                'Multi-GPU Training': False,
                'Quantization-Aware Training': False,
                'Advanced Augmentation': False,
                'Hyperparameter Optimization': False,
                'Adaptive Batch Sizing': False,
                'Model Ensemble': False,
                'Advanced Metrics': False
            },
            'P2_Nice_to_Have': {
                'Docker Containerization': False,
                'MLflow Integration': False,
                'Model Versioning': False,
                'A/B Testing': False,
                'Data Caching': False
            }
        }
        
        # Check for features based on file analysis
        for file_info in self.analysis['python_files']:
            functions = file_info.get('functions', [])
            classes = file_info.get('classes', [])
            
            # P0 checks
            if 'setup_reproducible_training' in functions:
                features['P0_Critical']['Reproducible Training'] = True
            if 'create_kfold_splits' in functions:
                features['P0_Critical']['K-Fold Cross-Validation'] = True
            if 'export_to_onnx' in functions:
                features['P0_Critical']['ONNX Export'] = True
            if 'export_to_tflite' in functions:
                features['P0_Critical']['TFLite Export'] = True
            if 'get_lr_scheduler' in functions:
                features['P0_Critical']['Advanced LR Schedulers'] = True
            
            # P1 checks
            if 'MultiGPUTrainer' in classes:
                features['P1_Important']['Multi-GPU Training'] = True
            if 'QuantizationAwareTraining' in classes:
                features['P1_Important']['Quantization-Aware Training'] = True
            if 'AdvancedAugmentationPipeline' in classes:
                features['P1_Important']['Advanced Augmentation'] = True
            if 'HyperparameterOptimizer' in classes:
                features['P1_Important']['Hyperparameter Optimization'] = True
            if 'AdaptiveBatchSizer' in classes:
                features['P1_Important']['Adaptive Batch Sizing'] = True
            if 'ModelEnsemble' in classes:
                features['P1_Important']['Model Ensemble'] = True
            if 'AdvancedMetrics' in classes:
                features['P1_Important']['Advanced Metrics'] = True
            
            # P2 checks
            if 'DockerfileGenerator' in classes:
                features['P2_Nice_to_Have']['Docker Containerization'] = True
            if 'MLflowExperimentTracker' in classes:
                features['P2_Nice_to_Have']['MLflow Integration'] = True
            if 'ModelRegistry' in classes:
                features['P2_Nice_to_Have']['Model Versioning'] = True
            if 'ABTestingFramework' in classes:
                features['P2_Nice_to_Have']['A/B Testing'] = True
            if 'DataCacheManager' in classes:
                features['P2_Nice_to_Have']['Data Caching'] = True
        
        return features
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        report = []
        report.append("=" * 80)
        report.append("PHOENIX PROTOCOL - COMPLETE FILE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        report.append("📊 PROJECT STATISTICS")
        report.append("-" * 80)
        report.append(f"Total Python Files: {len(self.analysis['python_files'])}")
        report.append(f"Total Documentation Files: {len(self.analysis['documentation'])}")
        report.append(f"Total Lines of Code: {self.analysis['total_lines']:,}")
        report.append(f"Total Functions: {self.analysis['total_functions']}")
        report.append(f"Total Classes: {self.analysis['total_classes']}")
        report.append(f"External Dependencies: {len(self.analysis['dependencies'])}")
        report.append("")
        
        # Documentation size
        total_doc_size = sum(d['size_kb'] for d in self.analysis['documentation'])
        report.append(f"Total Documentation: {total_doc_size:.1f} KB")
        report.append("")
        
        # File categories
        categories = self.categorize_files()
        report.append("📁 FILES BY CATEGORY")
        report.append("-" * 80)
        
        for category, files in categories.items():
            if not files:
                continue
            report.append(f"\n{category} ({len(files)} files):")
            for file_info in files:
                if 'lines' in file_info:
                    report.append(f"  • {Path(file_info['path']).name}")
                    report.append(f"    - Lines: {file_info['lines']}")
                    report.append(f"    - Functions: {file_info['function_count']}")
                    report.append(f"    - Classes: {file_info['class_count']}")
                else:
                    report.append(f"  • {Path(file_info['path']).name} ({file_info['size_kb']:.1f} KB)")
        
        report.append("")
        
        # Feature completeness
        features = self.check_feature_completeness()
        report.append("✅ FEATURE COMPLETENESS CHECK")
        report.append("-" * 80)
        
        for priority, feature_dict in features.items():
            implemented = sum(1 for v in feature_dict.values() if v)
            total = len(feature_dict)
            percentage = (implemented / total * 100) if total > 0 else 0
            
            report.append(f"\n{priority.replace('_', ' ')} Features: {implemented}/{total} ({percentage:.0f}%)")
            for feature, status in feature_dict.items():
                symbol = "✅" if status else "❌"
                report.append(f"  {symbol} {feature}")
        
        report.append("")
        
        # Dependencies
        report.append("📦 EXTERNAL DEPENDENCIES")
        report.append("-" * 80)
        major_deps = [d for d in self.analysis['dependencies'] 
                     if d in ['tensorflow', 'numpy', 'scipy', 'sklearn', 'optuna', 'mlflow', 'onnx']]
        for dep in sorted(major_deps):
            report.append(f"  • {dep}")
        report.append(f"  ... and {len(self.analysis['dependencies']) - len(major_deps)} more")
        report.append("")
        
        # Top files by size
        report.append("📈 LARGEST FILES (Top 10)")
        report.append("-" * 80)
        sorted_files = sorted(self.analysis['python_files'], 
                            key=lambda x: x.get('lines', 0), reverse=True)[:10]
        for i, file_info in enumerate(sorted_files, 1):
            report.append(f"  {i}. {Path(file_info['path']).name}")
            report.append(f"     {file_info.get('lines', 0)} lines, "
                        f"{file_info.get('function_count', 0)} functions, "
                        f"{file_info.get('class_count', 0)} classes")
        report.append("")
        
        # Most complex files
        report.append("🔧 MOST COMPLEX FILES (Top 5)")
        report.append("-" * 80)
        sorted_by_complexity = sorted(self.analysis['python_files'],
                                     key=lambda x: x.get('function_count', 0) + x.get('class_count', 0) * 2,
                                     reverse=True)[:5]
        for i, file_info in enumerate(sorted_by_complexity, 1):
            complexity = file_info.get('function_count', 0) + file_info.get('class_count', 0) * 2
            report.append(f"  {i}. {Path(file_info['path']).name}")
            report.append(f"     Complexity: {complexity} "
                        f"({file_info.get('class_count', 0)} classes, "
                        f"{file_info.get('function_count', 0)} functions)")
        report.append("")
        
        # Grade assessment
        report.append("🎓 OVERALL PROJECT GRADE")
        report.append("-" * 80)
        
        # Calculate grade
        p0_features = features['P0_Critical']
        p1_features = features['P1_Important']
        p2_features = features['P2_Nice_to_Have']
        
        p0_score = sum(1 for v in p0_features.values() if v) / len(p0_features) * 100
        p1_score = sum(1 for v in p1_features.values() if v) / len(p1_features) * 100
        p2_score = sum(1 for v in p2_features.values() if v) / len(p2_features) * 100
        
        # Weighted grade (P0: 50%, P1: 30%, P2: 20%)
        final_grade = (p0_score * 0.5 + p1_score * 0.3 + p2_score * 0.2)
        
        if final_grade >= 95:
            grade_letter = "A+"
        elif final_grade >= 90:
            grade_letter = "A"
        elif final_grade >= 85:
            grade_letter = "A-"
        elif final_grade >= 80:
            grade_letter = "B+"
        else:
            grade_letter = "B"
        
        report.append(f"P0 (Critical): {p0_score:.0f}%")
        report.append(f"P1 (Important): {p1_score:.0f}%")
        report.append(f"P2 (Nice-to-Have): {p2_score:.0f}%")
        report.append("")
        report.append(f"FINAL GRADE: {grade_letter} ({final_grade:.1f}/100)")
        report.append("")
        
        if final_grade >= 95:
            report.append("🏆 EXCELLENT - Production ready, all critical features implemented")
        elif final_grade >= 90:
            report.append("✅ VERY GOOD - Production ready, minor features missing")
        elif final_grade >= 85:
            report.append("👍 GOOD - Nearly production ready, some important features missing")
        else:
            report.append("⚠️  NEEDS WORK - Some critical features still missing")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, filepath: str = 'COMPLETE_FILE_ANALYSIS.md'):
        """Save analysis report to file."""
        report = self.generate_report()
        with open(filepath, 'w') as f:
            f.write(report)
        print(f"✅ Analysis report saved: {filepath}")
        return filepath


def run_analysis():
    """Run complete project analysis."""
    analyzer = ProjectAnalyzer()
    analyzer.analyze_all_files()
    
    # Generate and print report
    report = analyzer.generate_report()
    print(report)
    
    # Save report
    analyzer.save_report('COMPLETE_FILE_ANALYSIS.md')
    
    # Save raw data as JSON
    analysis_data = {
        'total_files': len(analyzer.analysis['python_files']),
        'total_docs': len(analyzer.analysis['documentation']),
        'total_lines': analyzer.analysis['total_lines'],
        'total_functions': analyzer.analysis['total_functions'],
        'total_classes': analyzer.analysis['total_classes'],
        'features': analyzer.check_feature_completeness()
    }
    
    with open('analysis_data.json', 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print("✅ Analysis data saved: analysis_data.json")


if __name__ == '__main__':
    run_analysis()
