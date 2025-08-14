#!/usr/bin/env python3
"""
Simple startup script for the Stock Prediction Dashboard
This script helps you start both the backend and frontend servers
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_python_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'flask', 'torch', 'numpy', 'pandas', 'yfinance', 
        'sklearn', 'nltk', 'textblob'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing Python packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r transformer_final/requirements.txt")
        return False
    
    print("✅ All Python dependencies are installed")
    return True

def check_node_dependencies():
    """Check if Node.js dependencies are installed"""
    frontend_dir = Path("frontend")
    node_modules = frontend_dir / "node_modules"
    
    if not node_modules.exists():
        print("❌ Node.js dependencies not installed")
        print("Please run: cd frontend && npm install")
        return False
    
    print("✅ Node.js dependencies are installed")
    return True

def start_backend():
    """Start the Flask backend server"""
    print("🚀 Starting Flask backend server...")
    
    backend_dir = Path("transformer_final")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return None
    
    try:
        # Start the Flask server using shell=True for better Windows compatibility
        process = subprocess.Popen(
            f'cd transformer_final && python enhanced_app.py',
            shell=True,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ Backend server started on http://localhost:5000")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Backend server failed to start: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def start_frontend():
    """Start the React frontend server"""
    print("🚀 Starting React frontend server...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return None
    
    try:
        # Start the React development server using shell=True for better Windows compatibility
        process = subprocess.Popen(
            f'cd frontend && npm start',
            shell=True,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for the server to start
        time.sleep(5)
        
        if process.poll() is None:
            print("✅ Frontend server started on http://localhost:3000")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Frontend server failed to start: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return None

def main():
    """Main function to start both servers"""
    print("🎯 Stock Prediction Dashboard Startup")
    print("=" * 50)
    
    # Check dependencies
    if not check_python_dependencies():
        return
    
    if not check_node_dependencies():
        return
    
    print("\n📋 Starting servers...")
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend server")
        return
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Failed to start frontend server")
        backend_process.terminate()
        return
    
    print("\n🎉 Both servers are running!")
    print("📊 Backend: http://localhost:5000")
    print("🌐 Frontend: http://localhost:3000")
    print("\n💡 Opening frontend in browser...")
    
    # Open the frontend in the default browser
    try:
        webbrowser.open("http://localhost:3000")
    except:
        print("⚠️  Could not open browser automatically")
    
    print("\n⏹️  Press Ctrl+C to stop both servers")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend server stopped unexpectedly")
                break
                
            if frontend_process.poll() is not None:
                print("❌ Frontend server stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        
        # Terminate both processes
        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()
        
        print("✅ Servers stopped")

if __name__ == "__main__":
    main() 