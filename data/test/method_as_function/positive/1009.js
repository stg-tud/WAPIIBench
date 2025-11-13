// This tests for escape sequences inside quotes
const axios = require('axios');
axios.post('https://petstore.swagger.io/v2/pets', {
    name: 'na\r\nme',
    tag: 'tag'
});