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
import { NotificationProvider } from "./pages/NotificationContext"; 
function App() {
  
  return (
  
    <Routes>
      {/* Public Routes */}
      <Route path="/" element={<Gate />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/login" element={<Login />} />

      {/* Protected Routes */}
      <Route
        path="/home"
        element={
          <NotificationProvider>  
          <ProtectedRoute>
              <Home />
          </ProtectedRoute>
          </NotificationProvider>  
        }
      />
      <Route
        path="/reports"
        element={
          <NotificationProvider>  
          <ProtectedRoute>
              <Reports />
          </ProtectedRoute>
          </NotificationProvider>  
        }
      />
      <Route
        path="/new-report"
        element={
          <NotificationProvider>  
          <ProtectedRoute>
              <NewReport />
          </ProtectedRoute>
          </NotificationProvider>  
        }
      />
      <Route
        path="/notifications"
        element={
          <NotificationProvider>  
          <ProtectedRoute>
              <Notifications />
          </ProtectedRoute>
          </NotificationProvider>  
        }
      />
      <Route
        path="/report/:id"
        element={
          <NotificationProvider>  
          <ProtectedRoute>
              <ReportRecords />
          </ProtectedRoute>
          </NotificationProvider>  
        }
      />
      <Route
        path="/about"
        element={
          <NotificationProvider>  
          <ProtectedRoute>
              <About />            
          </ProtectedRoute>
          </NotificationProvider>  
        }
      />
    </Routes>
  );
}

export default App;
