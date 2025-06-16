import { Alert } from '@mui/material';

const ErrorAlert = ({ message }) => {
  return (
    <Alert severity="error" sx={{ mb: 4 }}>
      {message}
    </Alert>
  );
};

export default ErrorAlert;