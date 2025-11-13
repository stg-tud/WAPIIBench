const axios = require('axios');
axios.get('https://petstore.swagger.io/v2/pets', {
    params: {
        tags: [1],
        limit: 10
    }
});