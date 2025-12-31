import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore
import stripe
import time
import datetime
import requests
import os

# -------------------- 配置部分 --------------------

# Firebase 配置
# TODO: 取消註解並填入正確的 Firebase 配置
# firebase_config = {
#     "apiKey": "YOUR_API_KEY",
#     "authDomain": "subtitlex-52615.firebaseapp.com",
#     "projectId": "subtitlex-52615",
#     "storageBucket": "subtitlex-52615.appspot.com",
#     "messagingSenderId": "656186755294",
#     "appId": "1:656186755294:web:29e06a0a6bb97eba78da4f",
#     "databaseURL": "https://subtitlex-d0142-default-rtdb.asia-southeast1.firebasedatabase.app",
# }

# 初始化 Firebase
# TODO: 取消註解以啟用 Firebase 功能
# firebase = pyrebase.initialize_app(firebase_config)
# auth = firebase.auth()

# Firebase Admin 初始化
# TODO: 設置環境變數 FIREBASE_CREDENTIALS_PATH 或將憑證文件放在專案根目錄
if not firebase_admin._apps:
    credentials_path = os.environ.get(
        'FIREBASE_CREDENTIALS_PATH',
        os.path.join(os.path.dirname(__file__), 'firebase-credentials.json')
    )
    if os.path.exists(credentials_path):
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred)
    else:
        st.warning(f"Firebase 憑證文件未找到: {credentials_path}")

db = firestore.client()

# Stripe 配置
stripe.api_key = "YOUR_STRIPE_SECRET_KEY"  # 请使用您的 Stripe Secret Key

# -------------------- 应用程序逻辑 --------------------

def signup():
    st.subheader("创建新账户")
    email = st.text_input("电子邮件", key="email_signup")
    password = st.text_input("密码", type="password", key="password_signup")
    if st.button("注册"):
        try:
            user = auth.create_user_with_email_and_password(email, password)
            st.success("注册成功！请登录。")
        except Exception as e:
            st.error(f"注册失败：{e}")

def login():
    st.subheader("用户登录")

    # 添加 Google 登录按钮
    if st.button("使用 Google 登录"):
        # 构建 Google OAuth 登录 URL
        oauth_url = (
            "https://accounts.google.com/o/oauth2/v2/auth"
            "?response_type=code"
            "&client_id=YOUR_GOOGLE_CLIENT_ID"
            "&redirect_uri=http://localhost:8501"
            "&scope=email%20profile%20openid"
            "&access_type=offline"
            "&prompt=consent"
        )
        st.markdown(f"[点击此处使用 Google 登录]({oauth_url})", unsafe_allow_html=True)

    # 处理重定向后的代码交换
    code = st.query_params.get('code')
    if code:
        # 交换代码获取访问令牌和 ID 令牌
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "code": code[0],
            "client_id": "YOUR_GOOGLE_CLIENT_ID",
            "client_secret": "YOUR_GOOGLE_CLIENT_SECRET",
            "redirect_uri": "http://localhost:8501",
            "grant_type": "authorization_code",
        }
        res = requests.post(token_url, data=data)
        if res.ok:
            tokens = res.json()
            id_token = tokens["id_token"]

            # 验证 ID 令牌
            decoded_token = auth.verify_id_token(id_token)
            user_email = decoded_token['email']

            st.session_state['logged_in'] = True
            st.session_state['user_email'] = user_email
            st.success(f"登录成功！欢迎，{user_email}")

            # 清除 URL 参数
            st.query_params.clear()
        else:
            st.error("登录失败，无法获取令牌。")


def logout():
    st.session_state['logged_in'] = False
    st.session_state['user_email'] = None
    st.success("已注销。")

def check_subscription_status():
    doc_ref = db.collection('users').document(st.session_state['user_email'])
    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict()
        st.session_state['is_subscribed'] = data.get('is_subscribed', False)
        st.session_state['usage_seconds'] = data.get('usage_seconds', 0)
        st.session_state['last_reset'] = data.get('last_reset')
    else:
        # 如果用户文档不存在，创建一个新的
        doc_ref.set({
            'is_subscribed': False,
            'usage_seconds': 0,
            'last_reset': datetime.datetime.now()
        })
        st.session_state['is_subscribed'] = False
        st.session_state['usage_seconds'] = 0
        st.session_state['last_reset'] = datetime.datetime.now()

def update_subscription_status(is_subscribed):
    doc_ref = db.collection('users').document(st.session_state['user_email'])
    doc_ref.update({
        'is_subscribed': is_subscribed,
        'usage_seconds': 0,
        'last_reset': datetime.datetime.now()
    })
    st.session_state['is_subscribed'] = is_subscribed
    st.session_state['usage_seconds'] = 0
    st.session_state['last_reset'] = datetime.datetime.now()

def reset_usage_if_new_month():
    last_reset = st.session_state.get('last_reset')
    if last_reset:
        last_reset_date = last_reset.date()
        current_date = datetime.date.today()
        if current_date.month != last_reset_date.month:
            # 重置使用时间
            st.session_state['usage_seconds'] = 0
            doc_ref = db.collection('users').document(st.session_state['user_email'])
            doc_ref.update({
                'usage_seconds': 0,
                'last_reset': datetime.datetime.now()
            })
    else:
        # 如果没有 last_reset，初始化它
        doc_ref = db.collection('users').document(st.session_state['user_email'])
        doc_ref.update({
            'last_reset': datetime.datetime.now()
        })

def create_checkout_session():
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': 'YOUR_PRICE_ID',  # 替换为您的价格 ID
                'quantity': 1,
            }],
            mode='subscription',
            success_url='http://localhost:8501?session_id={CHECKOUT_SESSION_ID}',
            cancel_url='http://localhost:8501',
            customer_email=st.session_state['user_email']
        )
        return session
    except Exception as e:
        st.error(f"创建支付会话失败：{e}")
        return None

def subscription_page():
    st.subheader("订阅服务")
    if st.button("订阅每月 $10"):
        session = create_checkout_session()
        if session:
            st.write("重定向到支付页面...")
            st.markdown(f"<a href='{session.url}' target='_self'>点击此处继续</a>", unsafe_allow_html=True)

def handle_payment_success():
    query_params = st.experimental_get_query_params()
    if 'session_id' in query_params:
        session_id = query_params['session_id'][0]
        session = stripe.checkout.Session.retrieve(session_id)
        if session.payment_status == 'paid':
            st.success("支付成功！")
            # 更新用户订阅状态
            update_subscription_status(True)
            # 清除 URL 参数
            st.query_params.clear()

def track_usage(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time

        # 更新使用时间到 Firestore
        st.session_state['usage_seconds'] += int(elapsed_time)
        doc_ref = db.collection('users').document(st.session_state['user_email'])
        doc_ref.update({
            'usage_seconds': st.session_state['usage_seconds']
        })

        return result
    return wrapper

@track_usage
def compress_audio(input_file):
    # 您的压缩音频代码
    st.write("正在压缩音频...")
    time.sleep(2)  # 模拟处理时间
    st.write("音频压缩完成！")

@track_usage
def transcribe_audio(input_file, output_file, language, prompt, api_key, temperature):
    # 您的转录音频代码
    st.write("正在转录音频...")
    time.sleep(5)  # 模拟处理时间
    st.write("音频转录完成！")

@track_usage
def translate_audio(input_file, output_file, prompt, api_key, temperature):
    # 您的翻译音频代码
    st.write("正在翻译音频...")
    time.sleep(3)  # 模拟处理时间
    st.write("音频翻译完成！")

def ai_subtitle_generator():
    st.write(f"欢迎回来，{st.session_state['user_email']}！")

    if not st.session_state.get('is_subscribed', False):
        st.warning("您还没有订阅，请先订阅。")
        subscription_page()
        return

    reset_usage_if_new_month()

    remaining_seconds = (20 * 3600) - st.session_state['usage_seconds']
    if remaining_seconds <= 0:
        st.error("您本月的20小时使用额度已用完。")
        return
    else:
        st.write(f"您本月还剩余 {remaining_seconds / 3600:.2f} 小时。")

    # 模拟音频处理流程
    if st.button("开始音频压缩"):
        compress_audio("input_file")
    if st.button("开始音频转录"):
        transcribe_audio("input_file", "output_file", "zh", "prompt", "api_key", 0.4)
    if st.button("开始音频翻译"):
        translate_audio("input_file", "output_file", "prompt", "api_key", 0.4)

def main():
    st.title("🚀 AI 生成字幕")

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['user_email'] = None

    if st.session_state['logged_in']:
        st.write(f"欢迎回来，{st.session_state['user_email']}！")
        # 在这里添加您的应用主功能
        # ...
        if st.sidebar.button("注销"):
            logout()
    else:
        login()

if __name__ == '__main__':
    main()
