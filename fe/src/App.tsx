import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import ProtectedRoute from "./components/ProtectedRoute";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import StreamPage from "./pages/StreamPage";
import CamerasPage from "./pages/CamerasPage";
import CameraRegisterPage from "./pages/CameraRegisterPage";
import EventLogsPage from "./pages/EventLogsPage";
import EventDetailPage from "./pages/EventDetailPage";
import UsersPage from "./pages/UsersPage";
import AdminDashboardPage from "./pages/AdminDashboardPage";

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/signup" element={<SignupPage />} />

          <Route path="/stream" element={<ProtectedRoute><StreamPage /></ProtectedRoute>} />
          <Route path="/events" element={<ProtectedRoute><EventLogsPage /></ProtectedRoute>} />
          <Route path="/events/:id" element={<ProtectedRoute><EventDetailPage /></ProtectedRoute>} />
          <Route path="/cameras" element={<ProtectedRoute><CamerasPage /></ProtectedRoute>} />
          <Route path="/cameras/register" element={<ProtectedRoute adminOnly><CameraRegisterPage /></ProtectedRoute>} />
          <Route path="/users" element={<ProtectedRoute adminOnly><UsersPage /></ProtectedRoute>} />
          <Route path="/admin" element={<ProtectedRoute adminOnly><AdminDashboardPage /></ProtectedRoute>} />

          <Route path="/" element={<Navigate to="/stream" replace />} />
          <Route path="*" element={<Navigate to="/stream" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
