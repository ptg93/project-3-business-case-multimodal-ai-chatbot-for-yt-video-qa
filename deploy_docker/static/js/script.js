async function processVideo() {
    const fileInput = document.getElementById('file-input');
    const processType = document.getElementById('process-type').value;
    const mode = document.getElementById('mode').value;
    
    if (!fileInput.files.length) {
        alert('Please select a video file to upload.');
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    formData.append('processType', processType);
    formData.append('mode', mode);

    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const progressContainer = document.getElementById('progress-container');

    progressContainer.style.display = 'block';
    progressBar.value = 0;
    progressText.textContent = 'Uploading and processing...';

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            if (result.message) {
                progressBar.value = 100;
                progressText.textContent = 'Processing complete. Chatbot is ready.';
                enableChatbot();
            } else {
                throw new Error('Failed to process video');
            }
        } else {
            throw new Error('Failed to process video');
        }
    } catch (error) {
        console.error('Error:', error);
        progressText.textContent = 'Error processing video.';
    }
}

function enableChatbot() {
    const chatInput = document.getElementById('chat-input');
    const chatButton = document.querySelector('#chat-form button');
    chatInput.disabled = false;
    chatButton.disabled = false;
}

async function sendMessage() {
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const message = chatInput.value.trim();

    if (!message) {
        alert('Please enter a message.');
        return;
    }

    chatMessages.innerHTML += `<div class="user-message"><strong>You:</strong> ${message}</div>`;
    chatInput.value = '';

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: message })
        });

        if (response.ok) {
            const data = await response.json();
            chatMessages.innerHTML += `<div class="bot-message"><strong>Bot:</strong> ${data.response}</div>`;
        } else {
            throw new Error('Failed to get response from chatbot');
        }
    } catch (error) {
        console.error('Error:', error);
        chatMessages.innerHTML += `<div class="bot-message"><strong>Bot:</strong> Error processing your message.</div>`;
    }

    chatMessages.scrollTop = chatMessages.scrollHeight;
}
