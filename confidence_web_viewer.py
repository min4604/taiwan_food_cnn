#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台灣美食 CNN 分類 - 信心分數網頁顯示器
Taiwan Food CNN Classification - Confidence Score Web Viewer

創建一個網頁應用程式，同時顯示圖片、信心分數和預測類別
"""

import os
import base64
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import csv

class ConfidenceWebViewer:
    """信心分數網頁顯示器"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.app.secret_key = 'taiwan_food_cnn_confidence_viewer'
        self.setup_routes()
        self.class_names = self.load_class_names()
    
    def load_class_names(self):
        """載入類別名稱"""
        class_file = 'archive/tw_food_101/tw_food_101_classes.csv'
        class_names = {}
        
        if os.path.exists(class_file):
            try:
                with open(class_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # 跳過標題行
                    for row in reader:
                        if len(row) >= 2:
                            class_id = int(row[0])
                            class_name = row[1]
                            class_names[class_id] = class_name
            except Exception as e:
                print(f"⚠️  載入類別名稱失敗: {e}")
        
        # 如果沒有類別檔案，使用預設名稱
        if not class_names:
            for i in range(101):
                class_names[i] = f"台灣美食類別 {i}"
        
        return class_names
    
    def load_prediction_results(self):
        """載入預測結果"""
        results_files = [
            'test_predictions_optimized_amd_npu.csv',
            'test_predictions_amd_npu.csv',
            'test_predictions.csv'
        ]
        
        results = []
        
        for results_file in results_files:
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            try:
                                result = {
                                    'id': int(row['Id']),
                                    'category': int(row['Category']),
                                    'confidence': float(row.get('Confidence', 0.0)),
                                    'path': row.get('Path', ''),
                                    'class_name': self.class_names.get(int(row['Category']), f"未知類別 {row['Category']}")
                                }
                                results.append(result)
                            except (ValueError, KeyError) as e:
                                continue
                    print(f"✅ 載入預測結果: {results_file} ({len(results)} 筆)")
                    break
                except Exception as e:
                    print(f"⚠️  載入 {results_file} 失敗: {e}")
                    continue
        
        if not results:
            print("❌ 找不到有效的預測結果檔案")
            return []
        
        # 按信心分數排序
        results.sort(key=lambda x: x['confidence'])
        return results
    
    def image_to_base64(self, image_path):
        """將圖片轉換為 base64 編碼"""
        try:
            if not os.path.exists(image_path):
                return None
            
            with open(image_path, 'rb') as f:
                image_data = f.read()
                base64_data = base64.b64encode(image_data).decode('utf-8')
                
                # 檢測圖片格式
                if image_path.lower().endswith(('.png', '.PNG')):
                    mime_type = 'image/png'
                elif image_path.lower().endswith(('.jpg', '.jpeg', '.JPG', '.JPEG')):
                    mime_type = 'image/jpeg'
                else:
                    mime_type = 'image/jpeg'  # 預設
                
                return f"data:{mime_type};base64,{base64_data}"
        except Exception as e:
            print(f"⚠️  圖片轉換失敗 {image_path}: {e}")
            return None
    
    def setup_routes(self):
        """設定路由"""
        
        @self.app.route('/')
        def index():
            """主頁面"""
            return render_template('confidence_viewer.html')
        
        @self.app.route('/api/results')
        def get_results():
            """取得預測結果 API"""
            try:
                results = self.load_prediction_results()
                
                # 取得查詢參數
                threshold = float(request.args.get('threshold', 0.5))
                limit = int(request.args.get('limit', 20))
                sort_by = request.args.get('sort', 'confidence')  # confidence, category, id
                
                # 篩選低信心圖片
                if threshold > 0:
                    filtered_results = [r for r in results if r['confidence'] > 0 and r['confidence'] < threshold]
                else:
                    filtered_results = results
                
                # 排序
                if sort_by == 'confidence':
                    filtered_results.sort(key=lambda x: x['confidence'])
                elif sort_by == 'category':
                    filtered_results.sort(key=lambda x: x['category'])
                elif sort_by == 'id':
                    filtered_results.sort(key=lambda x: x['id'])
                
                # 限制數量
                filtered_results = filtered_results[:limit]
                
                # 載入圖片
                for result in filtered_results:
                    result['image_data'] = self.image_to_base64(result['path'])
                    result['image_exists'] = result['image_data'] is not None
                
                return jsonify({
                    'success': True,
                    'total': len(results),
                    'filtered': len(filtered_results),
                    'results': filtered_results,
                    'threshold': threshold,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        @self.app.route('/api/stats')
        def get_stats():
            """取得統計資訊 API"""
            try:
                results = self.load_prediction_results()
                
                if not results:
                    return jsonify({'success': False, 'error': '沒有資料'})
                
                # 計算統計資訊
                confidences = [r['confidence'] for r in results if r['confidence'] > 0]
                
                stats = {
                    'total_images': len(results),
                    'valid_confidences': len(confidences),
                    'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
                    'min_confidence': min(confidences) if confidences else 0,
                    'max_confidence': max(confidences) if confidences else 0,
                    'low_confidence_count': len([c for c in confidences if c < 0.5]),
                    'medium_confidence_count': len([c for c in confidences if 0.5 <= c < 0.8]),
                    'high_confidence_count': len([c for c in confidences if c >= 0.8])
                }
                
                return jsonify({
                    'success': True,
                    'stats': stats
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
    
    def create_html_template(self):
        """創建 HTML 模板"""
        template_dir = 'templates'
        if not os.path.exists(template_dir):
            os.makedirs(template_dir)
        
        html_content = '''<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🍜 台灣美食 CNN - 信心分數檢視器</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #ff6b6b, #ffa500);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .controls {
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #dee2e6;
        }
        
        .control-group {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: center;
            justify-content: center;
        }
        
        .control-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 5px;
        }
        
        .control-item label {
            font-weight: bold;
            color: #495057;
        }
        
        .control-item input, .control-item select {
            padding: 8px 12px;
            border: 2px solid #dee2e6;
            border-radius: 5px;
            font-size: 14px;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .stats {
            background: #e3f2fd;
            padding: 15px;
            margin: 20px;
            border-radius: 10px;
            border-left: 5px solid #2196f3;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }
        
        .stat-item {
            text-align: center;
            background: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #1976d2;
        }
        
        .results {
            padding: 20px;
        }
        
        .loading {
            text-align: center;
            padding: 50px;
            font-size: 18px;
            color: #666;
        }
        
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .image-card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .image-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        .image-container {
            position: relative;
            height: 200px;
            overflow: hidden;
        }
        
        .image-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.3s;
        }
        
        .image-container:hover img {
            transform: scale(1.1);
        }
        
        .confidence-badge {
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            border-radius: 15px;
            color: white;
            font-weight: bold;
            font-size: 12px;
        }
        
        .confidence-very-low { background: #f44336; }
        .confidence-low { background: #ff9800; }
        .confidence-medium { background: #ffeb3b; color: #333; }
        .confidence-high { background: #4caf50; }
        
        .card-content {
            padding: 15px;
        }
        
        .card-title {
            font-size: 16px;
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
        }
        
        .card-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
            font-size: 14px;
            color: #666;
        }
        
        .category-tag {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
        }
        
        .no-results {
            text-align: center;
            padding: 50px;
            color: #666;
        }
        
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            margin: 20px;
            border-radius: 5px;
            border-left: 5px solid #f44336;
        }
        
        @media (max-width: 768px) {
            .control-group {
                flex-direction: column;
            }
            
            .image-grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🍜 台灣美食 CNN 分類系統</h1>
            <p>信心分數檢視器 - 分析模型預測結果</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <div class="control-item">
                    <label for="threshold">信心門檻值</label>
                    <input type="number" id="threshold" min="0" max="1" step="0.1" value="0.5">
                </div>
                <div class="control-item">
                    <label for="limit">顯示數量</label>
                    <input type="number" id="limit" min="1" max="100" value="20">
                </div>
                <div class="control-item">
                    <label for="sort">排序方式</label>
                    <select id="sort">
                        <option value="confidence">信心分數</option>
                        <option value="category">預測類別</option>
                        <option value="id">圖片編號</option>
                    </select>
                </div>
                <button class="btn" onclick="loadResults()">🔍 載入結果</button>
                <button class="btn" onclick="loadStats()">📊 顯示統計</button>
            </div>
        </div>
        
        <div id="stats" class="stats" style="display: none;">
            <h3>📊 預測統計資訊</h3>
            <div class="stats-grid" id="stats-content">
                <!-- 統計資訊將在這裡顯示 -->
            </div>
        </div>
        
        <div class="results">
            <div id="loading" class="loading">
                🔄 載入中，請稍候...
            </div>
            
            <div id="error" class="error" style="display: none;">
                <!-- 錯誤訊息將在這裡顯示 -->
            </div>
            
            <div id="results-info" style="display: none; margin-bottom: 20px; text-align: center; color: #666;">
                <!-- 結果資訊將在這裡顯示 -->
            </div>
            
            <div id="image-grid" class="image-grid">
                <!-- 圖片將在這裡顯示 -->
            </div>
            
            <div id="no-results" class="no-results" style="display: none;">
                <h3>😊 沒有找到符合條件的圖片</h3>
                <p>嘗試調整信心門檻值或增加顯示數量</p>
            </div>
        </div>
    </div>

    <script>
        let currentResults = [];
        
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('error').style.display = 'none';
            document.getElementById('image-grid').innerHTML = '';
            document.getElementById('no-results').style.display = 'none';
            document.getElementById('results-info').style.display = 'none';
        }
        
        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
        }
        
        function showError(message) {
            hideLoading();
            const errorDiv = document.getElementById('error');
            errorDiv.innerHTML = `❌ 錯誤: ${message}`;
            errorDiv.style.display = 'block';
        }
        
        function getConfidenceClass(confidence) {
            if (confidence < 0.3) return 'confidence-very-low';
            if (confidence < 0.5) return 'confidence-low';
            if (confidence < 0.8) return 'confidence-medium';
            return 'confidence-high';
        }
        
        function getConfidenceText(confidence) {
            if (confidence < 0.3) return '非常低';
            if (confidence < 0.5) return '低';
            if (confidence < 0.8) return '中等';
            return '高';
        }
        
        function displayResults(data) {
            hideLoading();
            
            const resultsInfo = document.getElementById('results-info');
            const imageGrid = document.getElementById('image-grid');
            const noResults = document.getElementById('no-results');
            
            if (data.results.length === 0) {
                noResults.style.display = 'block';
                return;
            }
            
            // 顯示結果資訊
            resultsInfo.innerHTML = `
                📊 找到 ${data.filtered} 張符合條件的圖片 (總共 ${data.total} 張)
                | 門檻值: ${data.threshold} | 更新時間: ${new Date(data.timestamp).toLocaleString()}
            `;
            resultsInfo.style.display = 'block';
            
            // 顯示圖片
            imageGrid.innerHTML = '';
            data.results.forEach(result => {
                const card = document.createElement('div');
                card.className = 'image-card';
                
                const confidenceClass = getConfidenceClass(result.confidence);
                const confidenceText = getConfidenceText(result.confidence);
                
                card.innerHTML = `
                    <div class="image-container">
                        ${result.image_exists ? 
                            `<img src="${result.image_data}" alt="圖片 ${result.id}" onerror="this.style.display='none'">` :
                            `<div style="display: flex; align-items: center; justify-content: center; height: 100%; background: #f5f5f5; color: #999;">圖片載入失敗</div>`
                        }
                        <div class="confidence-badge ${confidenceClass}">
                            ${confidenceText} ${result.confidence.toFixed(3)}
                        </div>
                    </div>
                    <div class="card-content">
                        <div class="card-title">圖片編號 ${result.id}</div>
                        <div class="card-info">
                            <span>預測類別: ${result.category}</span>
                            <span class="category-tag">${result.class_name}</span>
                        </div>
                        <div class="card-info">
                            <span>信心分數: ${result.confidence.toFixed(4)}</span>
                            <span style="font-size: 12px; color: #999;">${result.path}</span>
                        </div>
                    </div>
                `;
                
                imageGrid.appendChild(card);
            });
        }
        
        function loadResults() {
            showLoading();
            
            const threshold = document.getElementById('threshold').value;
            const limit = document.getElementById('limit').value;
            const sort = document.getElementById('sort').value;
            
            fetch(`/api/results?threshold=${threshold}&limit=${limit}&sort=${sort}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        currentResults = data.results;
                        displayResults(data);
                    } else {
                        showError(data.error);
                    }
                })
                .catch(error => {
                    showError(`網路錯誤: ${error.message}`);
                });
        }
        
        function loadStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        displayStats(data.stats);
                    } else {
                        showError(data.error);
                    }
                })
                .catch(error => {
                    showError(`統計載入失敗: ${error.message}`);
                });
        }
        
        function displayStats(stats) {
            const statsDiv = document.getElementById('stats');
            const statsContent = document.getElementById('stats-content');
            
            statsContent.innerHTML = `
                <div class="stat-item">
                    <div class="stat-value">${stats.total_images}</div>
                    <div>總圖片數</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${stats.avg_confidence.toFixed(3)}</div>
                    <div>平均信心</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${stats.low_confidence_count}</div>
                    <div>低信心圖片</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${stats.medium_confidence_count}</div>
                    <div>中等信心圖片</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${stats.high_confidence_count}</div>
                    <div>高信心圖片</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${stats.min_confidence.toFixed(3)}</div>
                    <div>最低信心</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${stats.max_confidence.toFixed(3)}</div>
                    <div>最高信心</div>
                </div>
            `;
            
            statsDiv.style.display = 'block';
        }
        
        // 頁面載入時自動載入結果
        window.onload = function() {
            loadResults();
            loadStats();
        };
        
        // 按 Enter 鍵載入結果
        document.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                loadResults();
            }
        });
    </script>
</body>
</html>'''
        
        template_file = os.path.join(template_dir, 'confidence_viewer.html')
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML 模板已創建: {template_file}")
    
    def run(self, host='127.0.0.1', port=5000, debug=True):
        """啟動網頁服務器"""
        self.create_html_template()
        
        print("🚀 啟動台灣美食 CNN 信心分數檢視器")
        print("=" * 50)
        print(f"📍 網址: http://{host}:{port}")
        print("💡 按 Ctrl+C 停止服務器")
        print("=" * 50)
        
        try:
            self.app.run(host=host, port=port, debug=debug)
        except KeyboardInterrupt:
            print("\n👋 服務器已停止")

def main():
    """主函數"""
    try:
        # 檢查 Flask 是否已安裝
        try:
            import flask
        except ImportError:
            print("❌ Flask 未安裝，正在安裝...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'flask'])
            print("✅ Flask 安裝完成")
        
        # 創建並啟動檢視器
        viewer = ConfidenceWebViewer()
        viewer.run()
        
    except Exception as e:
        print(f"❌ 啟動失敗: {e}")
        print("💡 請確認:")
        print("   1. Python 環境正常")
        print("   2. 有預測結果檔案")
        print("   3. 圖片路徑正確")

if __name__ == '__main__':
    main()