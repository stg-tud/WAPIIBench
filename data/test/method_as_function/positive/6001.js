// This tests Bearer authentication
const axios = require('axios');
const id = 42
const token = 'mySuperSecureToken';
axios.get(`https://petstore.swagger.io/v2/pets-secure/${id}`,{headers: {Authorization: `Bearer ${token}`}});