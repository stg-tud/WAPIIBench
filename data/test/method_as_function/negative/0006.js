const axios = require('axios');
axios.get('https://petstore.swagger.io/v2/pets', {
    params: {
        tags: ['tag1', 2],
        limit: 10
    }
});