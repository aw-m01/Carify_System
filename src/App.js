import { Routes, Route } from "react-router-dom";
import ProtectedRoute from "./pages/ProtectedRoute";
import Gate from "./pages/Gate";
import Signup from "./pages/Signup";
import Login from "./pages/Login";
import Home from "./pages/Home";
import Reports from "./pages/Reports";
import NewReport from "./pages/NewReport";
import Notifications from "./pages/Notifications";
import ReportRecords from "./pages/ReportRecords";
import About from "./pages/About";
import { NotificationProvider } from "./pages/NotificationContext"; // Adjust path as needed
function App() {
  
  return (
   <NotificationProvider> {/* Wrap the entire app with NotificationProvider */} 
    <Routes>
      {/* Public Routes */}
      <Route path="/" element={<Gate />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/login" element={<Login />} />

      {/* Protected Routes */}
      <Route
        path="/home"
        element={
          <ProtectedRoute>
              <Home />
          </ProtectedRoute>
        }
      />
      <Route
        path="/reports"
        element={
          <ProtectedRoute>
              <Reports />
          </ProtectedRoute>
        }
      />
      <Route
        path="/new-report"
        element={
          <ProtectedRoute>
              <NewReport />
          </ProtectedRoute>
        }
      />
      <Route
        path="/notifications"
        element={
          <ProtectedRoute>
              <Notifications />
          </ProtectedRoute>
        }
      />
      <Route
        path="/report/:id"
        element={
          <ProtectedRoute>
              <ReportRecords />
          </ProtectedRoute>
        }
      />
      <Route
        path="/about"
        element={
          <ProtectedRoute>
              <About />
            
          </ProtectedRoute>
        }
      />
    </Routes>
   </NotificationProvider>
  );
}

export default App;