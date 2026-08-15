import React from 'react';
import ReactDOM from 'react-dom/client';
import { AuthProvider } from './store/AuthContext';
import { ChatProvider } from './store/ChatContext';
import { TodoProvider } from './store/TodoContext';
import { PomodoroProvider } from './store/PomodoroContext';
import App from './App';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AuthProvider>
      <ChatProvider>
        <TodoProvider>
          <PomodoroProvider>
            <App />
          </PomodoroProvider>
        </TodoProvider>
      </ChatProvider>
    </AuthProvider>
  </React.StrictMode>
);
