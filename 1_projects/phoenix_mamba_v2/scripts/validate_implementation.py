"""
Lightweight Validation Script for Phoenix Protocol
Validates code structure, syntax, and logic without requiring heavy dependencies.
"""

import os
import sys
import ast

def validate_python_file(filepath):
    """Validate Python file for syntax errors and basic structure."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        
        # Parse the file
        tree = ast.parse(code, filename=filepath)
        
        # Count functions and classes
        functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        
        return True, functions, classes, None
    except SyntaxError as e:
        return False, 0, 0, str(e)
    except Exception as e:
        return False, 0, 0, str(e)

def check_import_consistency(filepath):
    """Check if imports in a file are consistent and properly structured."""
    issues = []
    
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        
        tree = ast.parse(code)
        
        # Check for relative imports that might cause issues
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    # Absolute import
                    if not node.module.startswith(('tensorflow', 'keras', 'numpy', 'scipy', 'cv2', 
                                                   'PIL', 'sklearn', 'matplotlib', 'pandas',
                                                   'models.', 'src.', 'config', 'imagehash')):
                        # Check if it's a local module without proper path
                        if '.' not in node.module:
                            issues.append(f"Potential import issue: 'from {node.module} import ...' should use package path")
        
        return issues
    except Exception as e:
        return [f"Error checking imports: {e}"]

def main():
    print("="*80)
    print("PHOENIX PROTOCOL - CODE VALIDATION")
    print("="*80)
    print()
    
    # Files to validate
    files_to_check = [
        ('models/dynamic_snake_conv.py', 'Dynamic Snake Convolution'),
        ('models/neurosnake_model.py', 'NeuroSnake Architecture'),
        ('models/cnn_model.py', 'Baseline CNN Model'),
        ('src/phoenix_optimizer.py', 'Adan Optimizer & Focal Loss'),
        ('src/physics_informed_augmentation.py', 'Physics-Informed Augmentation'),
        ('src/data_deduplication.py', 'Data Deduplication'),
        ('src/train_phoenix.py', 'Training Pipeline'),
        ('src/int8_quantization.py', 'INT8 Quantization'),
        ('src/comparative_analysis.py', 'Comparative Analysis'),
        ('config.py', 'Configuration'),
    ]
    
    print("1. SYNTAX VALIDATION")
    print("-" * 80)
    
    total_files = 0
    valid_files = 0
    total_functions = 0
    total_classes = 0
    
    for filepath, description in files_to_check:
        if os.path.exists(filepath):
            valid, funcs, classes, error = validate_python_file(filepath)
            total_files += 1
            
            if valid:
                valid_files += 1
                total_functions += funcs
                total_classes += classes
                print(f"✓ {description:40} ({funcs} functions, {classes} classes)")
            else:
                print(f"✗ {description:40} ERROR: {error}")
        else:
            print(f"✗ {description:40} FILE NOT FOUND")
    
    print()
    print("2. IMPORT CONSISTENCY CHECK")
    print("-" * 80)
    
    import_issues_found = False
    for filepath, description in files_to_check:
        if os.path.exists(filepath):
            issues = check_import_consistency(filepath)
            if issues:
                import_issues_found = True
                print(f"⚠ {description}:")
                for issue in issues:
                    print(f"    {issue}")
    
    if not import_issues_found:
        print("✓ All imports appear consistent")
    
    print()
    print("3. DOCUMENTATION CHECK")
    print("-" * 80)
    
    docs = [
        ('README.md', 'Main README'),
        ('PHOENIX_PROTOCOL.md', 'Phoenix Protocol Guide'),
        ('SECURITY_ANALYSIS.md', 'Security Analysis'),
        ('requirements.txt', 'Dependencies'),
        ('Research_Paper_Brain_Tumor_Detection.md', 'Research Paper'),
    ]
    
    docs_found = 0
    for doc, description in docs:
        if os.path.exists(doc):
            size = os.path.getsize(doc)
            print(f"✓ {description:40} ({size:,} bytes)")
            docs_found += 1
        else:
            print(f"✗ {description:40} NOT FOUND")
    
    print()
    print("4. KEY FEATURES VERIFICATION")
    print("-" * 80)
    
    # Check for key implementation features in files
    features = [
        ('models/dynamic_snake_conv.py', ['DynamicSnakeConv2D', '_deformable_conv2d', '_bilinear_sample'], 
         'Dynamic Snake Convolution implementation'),
        ('src/phoenix_optimizer.py', ['AdanOptimizer', 'FocalLoss', '_resource_apply_dense'],
         'Adan optimizer and Focal Loss'),
        ('src/physics_informed_augmentation.py', ['elastic_deformation', 'rician_noise', 'PhysicsInformedAugmentation'],
         'Physics-informed augmentation'),
        ('src/data_deduplication.py', ['ImageDeduplicator', 'compute_phash', 'detect_cross_split_duplicates'],
         'pHash-based deduplication'),
    ]
    
    for filepath, keywords, description in features:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read()
            
            found_keywords = [kw for kw in keywords if kw in content]
            if len(found_keywords) == len(keywords):
                print(f"✓ {description}")
            else:
                missing = set(keywords) - set(found_keywords)
                print(f"⚠ {description} (missing: {', '.join(missing)})")
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Files validated: {valid_files}/{total_files}")
    print(f"Total functions: {total_functions}")
    print(f"Total classes: {total_classes}")
    print(f"Documentation files: {docs_found}/{len(docs)}")
    print()
    
    if valid_files == total_files and docs_found == len(docs):
        print("✓ All validations passed!")
        print("  The Phoenix Protocol implementation is structurally sound.")
        print("  Note: Full functional testing requires TensorFlow and other dependencies.")
        return True
    else:
        print("⚠ Some validations failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
