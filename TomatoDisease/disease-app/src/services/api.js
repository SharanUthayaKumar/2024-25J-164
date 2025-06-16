import axios from 'axios';

const API_URL = 'http://localhost:5000/predict';


export const predictDisease = async (file) => {
    const formData = new FormData();
    formData.append('file',file);

    try{
        const response = await axios.post(API_URL,formData,{
            headers:{
                'Content-Type'  : 'multipart/form-data',
            }
        });
        return response.data;

    }catch(error){
        throw new Error(error.response?.data?.error || 'failed to connect to the server')
    }
}