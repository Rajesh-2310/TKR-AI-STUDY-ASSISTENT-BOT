from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import logging
from werkzeug.utils import secure_filename

from config import Config
from database import get_db, init_db
from models import Subject, Material, Syllabus, ImportantQuestion, ChatHistory, ExtractedImage
from pdf_processor import PDFProcessor
from gemini_rag import get_gemini_rag_engine  # Using Gemini AI-powered RAG engine
from auth import AuthService  # Admin authentication
from email_service import email_service  # Email verification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize directories
Config.init_app()

# Initialize email service
email_service.init_app(app)

# Initialize processors
pdf_processor = PDFProcessor()

# Initialize Gemini RAG engine at startup (not lazy loaded)
logger.info("Initializing Gemini RAG engine...")
gemini_rag_engine = get_gemini_rag_engine()
logger.info("Gemini RAG engine ready")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'TKR Chatbot API is running'})


@app.route('/api/subjects', methods=['GET'])
def get_subjects():
    """Get all subjects or filter by semester"""
    try:
        semester = request.args.get('semester', type=int)
        
        if semester:
            subjects = Subject.get_by_semester(semester)
        else:
            subjects = Subject.get_all()
        
        return jsonify({'success': True, 'subjects': subjects})
    except Exception as e:
        logger.error(f"Error fetching subjects: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/subjects', methods=['POST'])
def create_subject():
    """Create new subject"""
    try:
        data = request.json
        subject_id = Subject.create(
            data['subject_code'],
            data['subject_name'],
            data['semester'],
            data['department']
        )
        return jsonify({'success': True, 'subject_id': subject_id})
    except Exception as e:
        logger.error(f"Error creating subject: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/materials', methods=['GET'])
def get_materials():
    """Get materials, optionally filtered by subject"""
    try:
        subject_id = request.args.get('subject_id', type=int)
        
        if subject_id:
            materials = Material.get_by_subject(subject_id)
        else:
            materials = Material.get_all()
        
        return jsonify({'success': True, 'materials': materials})
    except Exception as e:
        logger.error(f"Error fetching materials: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_material():
    """Upload and process PDF material"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Only PDF allowed'}), 400
        
        # Get form data
        subject_id = request.form.get('subject_id', type=int)
        title = request.form.get('title', '')
        description = request.form.get('description', '')
        
        if not subject_id:
            return jsonify({'success': False, 'error': 'Subject ID required'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        
        file_size = os.path.getsize(file_path)
        
        # Create material record
        material_id = Material.create(
            subject_id,
            title or filename,
            description,
            file_path,
            'pdf',
            file_size
        )
        
        # Process PDF in background
        try:
            logger.info(f"Processing PDF for material {material_id}")
            result = pdf_processor.process_pdf(file_path, material_id)
            
            # Generate and store AI embeddings with Gemini RAG (engine already initialized)
            gemini_rag_engine.store_embeddings(material_id, result['chunks'])
            
            # Store extracted images
            for img in result['images']:
                ExtractedImage.create(
                    material_id,
                    img['path'],
                    img['page'],
                    img['type'],
                    None
                )
            
            # Mark as processed
            Material.mark_processed(material_id)
            
            logger.info(f"Successfully processed material {material_id}")
            
            return jsonify({
                'success': True,
                'material_id': material_id,
                'chunks_created': len(result['chunks']),
                'images_extracted': len(result['images'])
            })
            
        except Exception as process_error:
            logger.error(f"Error processing PDF: {process_error}")
            return jsonify({
                'success': True,
                'material_id': material_id,
                'warning': 'File uploaded but processing failed',
                'error': str(process_error)
            })
        
    except Exception as e:
        logger.error(f"Error uploading material: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/materials/<int:material_id>/download', methods=['GET'])
def download_material(material_id):
    """Download material file"""
    try:
        material = Material.get_by_id(material_id)
        if not material:
            return jsonify({'success': False, 'error': 'Material not found'}), 404
        
        return send_file(material['file_path'], as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading material: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/materials/<int:material_id>/images', methods=['GET'])
def get_material_images(material_id):
    """Get extracted images from material"""
    try:
        images = ExtractedImage.get_by_material(material_id)
        return jsonify({'success': True, 'images': images})
    except Exception as e:
        logger.error(f"Error fetching images: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/syllabus', methods=['GET'])
def get_syllabus():
    """Get syllabus by subject"""
    try:
        subject_id = request.args.get('subject_id', type=int)
        
        if not subject_id:
            return jsonify({'success': False, 'error': 'Subject ID required'}), 400
        
        syllabus = Syllabus.get_by_subject(subject_id)
        return jsonify({'success': True, 'syllabus': syllabus})
    except Exception as e:
        logger.error(f"Error fetching syllabus: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/syllabus', methods=['POST'])
def create_syllabus():
    """Create syllabus entry"""
    try:
        data = request.json
        syllabus_id = Syllabus.create(
            data['subject_id'],
            data['unit_number'],
            data['unit_name'],
            data['topics'],
            data.get('learning_outcomes', '')
        )
        return jsonify({'success': True, 'syllabus_id': syllabus_id})
    except Exception as e:
        logger.error(f"Error creating syllabus: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/important-questions', methods=['GET'])
def get_important_questions():
    """Get important questions"""
    try:
        subject_id = request.args.get('subject_id', type=int)
        question_type = request.args.get('type')
        
        if subject_id:
            questions = ImportantQuestion.get_by_subject(subject_id, question_type)
        else:
            questions = ImportantQuestion.get_all()
        
        return jsonify({'success': True, 'questions': questions})
    except Exception as e:
        logger.error(f"Error fetching questions: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/important-questions', methods=['POST'])
def create_important_question():
    """Create important question"""
    try:
        data = request.json
        question_id = ImportantQuestion.create(
            data['subject_id'],
            data['question'],
            data.get('answer', ''),
            data.get('question_type', 'short'),
            data.get('difficulty', 'medium'),
            data.get('unit_number')
        )
        return jsonify({'success': True, 'question_id': question_id})
    except Exception as e:
        logger.error(f"Error creating question: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages and answer questions"""
    try:
        data = request.json
        message = data.get('message', '').strip()
        subject_id = data.get('subject_id')
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not message:
            return jsonify({'success': False, 'error': 'Message required'}), 400
        
        # Get answer using Gemini AI-powered RAG (engine already initialized at startup)
        result = gemini_rag_engine.answer_question(message, subject_id)
        
        # Save to chat history
        context_info = {
            'sources': result.get('sources', []),
            'confidence': result.get('confidence', 0)
        }
        
        ChatHistory.create(
            session_id,
            message,
            result['answer'],
            str(context_info)
        )
        
        return jsonify({
            'success': True,
            'answer': result['answer'],
            'sources': result.get('sources', []),
            'confidence': result.get('confidence', 0),
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Get chat history for a session"""
    try:
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'Session ID required'}), 400
        
        history = ChatHistory.get_by_session(session_id)
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500




@app.route('/api/upload/syllabus', methods=['POST'])
def upload_syllabus():
    """Upload syllabus entry"""
    try:
        data = request.get_json()
        
        subject_id = data.get('subject_id')
        unit_number = data.get('unit_number')
        unit_name = data.get('unit_name')
        topics = data.get('topics')
        learning_outcomes = data.get('learning_outcomes', '')
        
        if not all([subject_id, unit_number, unit_name, topics]):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        syllabus_id = Syllabus.create(subject_id, unit_number, unit_name, topics, learning_outcomes)
        
        logger.info(f"Syllabus created: {syllabus_id}")
        return jsonify({'success': True, 'syllabus_id': syllabus_id}), 201
        
    except Exception as e:
        logger.error(f"Error uploading syllabus: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/upload/question', methods=['POST'])
def upload_question():
    """Upload important question"""
    try:
        data = request.get_json()
        
        subject_id = data.get('subject_id')
        question = data.get('question')
        answer = data.get('answer', '')
        question_type = data.get('question_type', 'short')
        difficulty = data.get('difficulty', 'medium')
        unit_number = data.get('unit_number')
        
        if not all([subject_id, question]):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        question_id = ImportantQuestion.create(
            subject_id, question, answer, question_type, difficulty, unit_number
        )
        
        logger.info(f"Important question created: {question_id}")
        return jsonify({'success': True, 'question_id': question_id}), 201
        
    except Exception as e:
        logger.error(f"Error uploading question: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== ADMIN AUTHENTICATION ROUTES ====================

@app.route('/api/admin/register', methods=['POST'])
def admin_register():
    """Register a new admin user"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'success': False, 'error': 'Email and password required'}), 400
        
        # Create admin user
        result = AuthService.create_admin_user(email, password)
        
        if result['success']:
            return jsonify(result), 201
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error in admin registration: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/admin/send-verification', methods=['POST'])
def send_verification():
    """Send verification code to email"""
    try:
        data = request.get_json()
        email = data.get('email')
        
        if not email:
            return jsonify({'success': False, 'error': 'Email required'}), 400
        
        # Create verification code
        result = AuthService.create_verification_code(email)
        
        if not result['success']:
            return jsonify(result), 400
        
        # Send email
        code = result['code']
        email_result = email_service.send_verification_code(email, code)
        
        return jsonify(email_result), 200
            
    except Exception as e:
        logger.error(f"Error sending verification: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/admin/verify-email', methods=['POST'])
def verify_email():
    """Verify email with code"""
    try:
        data = request.get_json()
        email = data.get('email')
        code = data.get('code')
        
        if not email or not code:
            return jsonify({'success': False, 'error': 'Email and code required'}), 400
        
        # Verify code
        result = AuthService.verify_code(email, code)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error verifying email: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    """Login admin user"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'success': False, 'error': 'Email and password required'}), 400
        
        # Login
        result = AuthService.login(email, password)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 401
            
    except Exception as e:
        logger.error(f"Error in admin login: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/admin/logout', methods=['POST'])
def admin_logout():
    """Logout admin user"""
    try:
        data = request.get_json()
        session_token = data.get('session_token')
        
        if not session_token:
            return jsonify({'success': False, 'error': 'Session token required'}), 400
        
        # Logout
        result = AuthService.logout(session_token)
        return jsonify(result), 200
            
    except Exception as e:
        logger.error(f"Error in admin logout: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/admin/check-auth', methods=['POST'])
def check_auth():
    """Check if session is valid"""
    try:
        data = request.get_json()
        session_token = data.get('session_token')
        
        if not session_token:
            return jsonify({'success': False, 'authenticated': False}), 200
        
        # Verify session
        result = AuthService.verify_session(session_token)
        
        if result['success']:
            return jsonify({
                'success': True,
                'authenticated': True,
                'email': result['email']
            }), 200
        else:
            return jsonify({'success': False, 'authenticated': False}), 200
            
    except Exception as e:
        logger.error(f"Error checking auth: {e}")
        return jsonify({'success': False, 'authenticated': False}), 200


# ==================== ADMIN-ONLY RESOURCE MANAGEMENT ====================

def require_admin(f):
    """Decorator to require admin authentication"""
    from functools import wraps
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get session token from header
        session_token = request.headers.get('X-Session-Token')
        
        if not session_token:
            return jsonify({'success': False, 'error': 'Authentication required'}), 401
        
        # Verify session
        result = AuthService.verify_session(session_token)
        
        if not result['success']:
            return jsonify({'success': False, 'error': 'Invalid or expired session'}), 401
        
        # Add admin info to request
        request.admin_id = result['admin_id']
        request.admin_email = result['email']
        
        return f(*args, **kwargs)
    
    return decorated_function


@app.route('/api/admin/materials/<int:material_id>', methods=['DELETE'])
@require_admin
def delete_material(material_id):
    """Delete a material (admin only)"""
    try:
        db = get_db()
        
        # Get material file path
        material = db.execute_query(
            "SELECT file_path FROM materials WHERE id = %s",
            (material_id,)
        )
        
        if not material or len(material) == 0:
            return jsonify({'success': False, 'error': 'Material not found'}), 404
        
        # Delete file if exists
        file_path = material[0]['file_path']
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
            except Exception as file_error:
                logger.warning(f"Could not delete file {file_path}: {file_error}")
                # Continue with database deletion even if file deletion fails
        
        # Delete from database (cascades to embeddings and images)
        try:
            db.execute_query(
                "DELETE FROM materials WHERE id = %s",
                (material_id,),
                fetch=False
            )
        except Exception as db_error:
            logger.error(f"Database error deleting material {material_id}: {db_error}")
            return jsonify({'success': False, 'error': f'Database error: {str(db_error)}'}), 500
        
        logger.info(f"Material {material_id} deleted by admin {request.admin_email}")
        return jsonify({'success': True, 'message': 'Material deleted successfully'}), 200
        
    except Exception as e:
        logger.error(f"Error deleting material: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/admin/syllabus/<int:syllabus_id>', methods=['DELETE'])
@require_admin
def delete_syllabus(syllabus_id):
    """Delete a syllabus (admin only)"""
    try:
        db = get_db()
        
        # Check if syllabus exists
        syllabus = db.execute_query(
            "SELECT id FROM syllabus WHERE id = %s",
            (syllabus_id,)
        )
        
        if not syllabus or len(syllabus) == 0:
            return jsonify({'success': False, 'error': 'Syllabus not found'}), 404
        
        # Delete from database (syllabus is text-only, no files to delete)
        try:
            db.execute_query(
                "DELETE FROM syllabus WHERE id = %s",
                (syllabus_id,),
                fetch=False
            )
        except Exception as db_error:
            logger.error(f"Database error deleting syllabus {syllabus_id}: {db_error}")
            return jsonify({'success': False, 'error': f'Database error: {str(db_error)}'}), 500
        
        logger.info(f"Syllabus {syllabus_id} deleted by admin {request.admin_email}")
        return jsonify({'success': True, 'message': 'Syllabus deleted successfully'}), 200
        
    except Exception as e:
        logger.error(f"Error deleting syllabus: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/admin/questions/<int:question_id>', methods=['DELETE'])
@require_admin
def delete_question(question_id):
    """Delete an important question (admin only)"""
    try:
        db = get_db()
        
        # Delete from database
        db.execute_query(
            "DELETE FROM important_questions WHERE id = %s",
            (question_id,),
            fetch=False
        )
        
        logger.info(f"Question {question_id} deleted by admin {request.admin_email}")
        return jsonify({'success': True, 'message': 'Question deleted successfully'}), 200
        
    except Exception as e:
        logger.error(f"Error deleting question: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== DATABASE INITIALIZATION ====================

@app.route('/api/init-db', methods=['POST'])
def initialize_database():
    """Initialize database (for setup only)"""
    try:
        success = init_db()
        if success:
            return jsonify({'success': True, 'message': 'Database initialized successfully'})
        else:
            return jsonify({'success': False, 'error': 'Database initialization failed'}), 500
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    # Connect to database
    try:
        get_db()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=Config.DEBUG)
