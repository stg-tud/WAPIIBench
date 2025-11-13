const axios = require('axios');
axios.post('https://petstore.swagger.io/v2/pets', {
    params: {
        name: 'name',
        tag: 'tag'
    }
});