// This tests API key authentication when multiple schemes are accepted
const axios = require('axios');
const id = 42
const token = 'mySuperSecureToken';
axios.put(`https://petstore.swagger.io/v2/pets-secure/${id}`,
    {
        name: 'name',
        tag: 'tag'
    },
    {params: {api_key: `Bearer ${token}`}});