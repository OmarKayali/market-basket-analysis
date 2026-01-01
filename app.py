from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)

# =====================================================
# LOAD DATA FROM JSON FILES
# =====================================================
def load_json(filename):
    """Load JSON file from output folder"""
    filepath = os.path.join('output', filename)
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Run data_processor.py first!")
        return None

# Load all data at startup
print("Loading data...")
statistics = load_json('statistics.json')
itemsets = load_json('itemsets.json')
rules = load_json('rules.json')
clusters = load_json('clusters.json')
products_data = load_json('products.json')
cluster_products = load_json('cluster_products.json')

products = products_data['products'] if products_data else []

print(f"Loaded: {len(itemsets or [])} itemsets, {len(rules or [])} rules, {len(clusters or [])} cluster points")

# =====================================================
# API ENDPOINTS
# =====================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'data_loaded': all([statistics, itemsets, rules, clusters, products])
    })

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get overall statistics"""
    if not statistics:
        return jsonify({'status': 'error', 'message': 'Statistics not loaded'}), 500
    
    return jsonify({
        'status': 'success',
        'stats': statistics
    })

@app.route('/api/itemsets', methods=['GET'])
def get_itemsets():
    """Get frequent itemsets with pagination"""
    if not itemsets:
        return jsonify({'status': 'error', 'message': 'Itemsets not loaded'}), 500
    
    # Get pagination parameters
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 50, type=int)
    search = request.args.get('search', '', type=str).lower()
    min_support = request.args.get('min_support', 0, type=float)
    
    # Filter itemsets
    filtered = itemsets
    
    if search:
        filtered = [item for item in filtered 
                   if any(search in i.lower() for i in item['items'])]
    
    if min_support > 0:
        filtered = [item for item in filtered if item['support'] >= min_support]
    
    # Sort by support (descending)
    filtered = sorted(filtered, key=lambda x: x['support'], reverse=True)
    
    # Paginate
    start = (page - 1) * limit
    end = start + limit
    paginated = filtered[start:end]
    
    return jsonify({
        'status': 'success',
        'data': paginated,
        'total': len(filtered),
        'page': page,
        'total_pages': (len(filtered) + limit - 1) // limit
    })

@app.route('/api/rules', methods=['GET'])
def get_rules():
    """Get association rules with filtering"""
    if not rules:
        return jsonify({'status': 'error', 'message': 'Rules not loaded'}), 500
    
    # Get filter parameters
    min_confidence = request.args.get('min_confidence', 0, type=float)
    min_lift = request.args.get('min_lift', 0, type=float)
    search = request.args.get('search', '', type=str).lower()
    
    # Filter rules
    filtered = rules
    
    if min_confidence > 0:
        filtered = [r for r in filtered if r['confidence'] >= min_confidence]
    
    if min_lift > 0:
        filtered = [r for r in filtered if r['lift'] >= min_lift]
    
    if search:
        filtered = [r for r in filtered 
                   if any(search in i.lower() for i in r['antecedent'] + r['consequent'])]
    
    # Sort by lift (descending)
    filtered = sorted(filtered, key=lambda x: x['lift'], reverse=True)
    
    return jsonify({
        'status': 'success',
        'data': filtered,
        'total': len(filtered)
    })

@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    """Get K-Means clustering results"""
    if not clusters:
        return jsonify({'status': 'error', 'message': 'Clusters not loaded'}), 500
    
    # Calculate cluster statistics
    cluster_stats = {}
    for point in clusters:
        cluster_id = str(point['cluster'])
        if cluster_id not in cluster_stats:
            cluster_stats[cluster_id] = []
        cluster_stats[cluster_id].append(point)
    
    # Calculate averages
    stats_summary = {}
    for cluster_id, points in cluster_stats.items():
        stats_summary[cluster_id] = {
            'count': len(points),
            'avg_pca_x': sum(p['pca_x'] for p in points) / len(points),
            'avg_pca_y': sum(p['pca_y'] for p in points) / len(points)
        }
    
    return jsonify({
        'status': 'success',
        'data': clusters,
        'cluster_stats': stats_summary
    })

@app.route('/api/cluster/<int:cluster_id>/products', methods=['GET'])
def get_cluster_products(cluster_id):
    """Get top products for a specific cluster"""
    if not cluster_products:
        return jsonify({'status': 'error', 'message': 'Cluster products not loaded'}), 500
    
    cluster_key = str(cluster_id)
    if cluster_key not in cluster_products:
        return jsonify({'status': 'error', 'message': f'Cluster {cluster_id} not found'}), 404
    
    return jsonify({
        'status': 'success',
        'cluster': cluster_id,
        'products': cluster_products[cluster_key]
    })

@app.route('/api/products', methods=['GET'])
def get_products():
    """Get list of all products"""
    if not products:
        return jsonify({'status': 'error', 'message': 'Products not loaded'}), 500
    
    # Check if we should filter for recommendable products only
    recommendable_only = request.args.get('recommendable', 'false').lower() == 'true'
    
    if recommendable_only and rules:
        # Get unique products that appear in antecedents (can generate recommendations)
        recommendable_products = set()
        for rule in rules:
            for item in rule['antecedent']:
                recommendable_products.add(item)
        
        # Return sorted list
        filtered_products = sorted(list(recommendable_products))
        
        return jsonify({
            'status': 'success',
            'products': filtered_products,
            'total': len(filtered_products),
            'filtered': True
        })
    
    return jsonify({
        'status': 'success',
        'products': products,
        'total': len(products),
        'filtered': False
    })

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Generate product recommendations based on selected products"""
    if not rules:
        return jsonify({'status': 'error', 'message': 'Rules not loaded'}), 500
    
    data = request.json
    selected_products = data.get('products', [])
    
    if not selected_products:
        return jsonify({
            'status': 'error',
            'message': 'No products provided'
        }), 400
    
    # Find rules where antecedent contains any selected product
    recommendations = {}
    
    for rule in rules:
        # Check if any selected product is in antecedent
        if any(prod in rule['antecedent'] for prod in selected_products):
            # Add consequents as recommendations
            for item in rule['consequent']:
                if item not in selected_products:
                    if item not in recommendations:
                        recommendations[item] = {
                            'item': item,
                            'confidence': rule['confidence'],
                            'lift': rule['lift'],
                            'support': rule['support'],
                            'count': 1
                        }
                    else:
                        # Keep highest confidence
                        if rule['confidence'] > recommendations[item]['confidence']:
                            recommendations[item]['confidence'] = rule['confidence']
                            recommendations[item]['lift'] = rule['lift']
                            recommendations[item]['support'] = rule['support']
                        recommendations[item]['count'] += 1
    
    # Convert to list and sort by confidence
    rec_list = list(recommendations.values())
    rec_list.sort(key=lambda x: (x['confidence'], x['lift']), reverse=True)
    
    # Take top 10
    rec_list = rec_list[:10]
    
    return jsonify({
        'status': 'success',
        'selected_products': selected_products,
        'recommendations': rec_list
    })

# =====================================================
# RUN SERVER
# =====================================================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("MBA Dashboard Backend Server")
    print("="*50)
    print("Server running on http://localhost:5001")
    print("API endpoints:")
    print("  GET  /api/health")
    print("  GET  /api/statistics")
    print("  GET  /api/itemsets")
    print("  GET  /api/rules")
    print("  GET  /api/clusters")
    print("  GET  /api/cluster/<id>/products")
    print("  GET  /api/products")
    print("  POST /api/recommendations")
    print("="*50 + "\n")
    
    app.run(debug=True, port=5001)