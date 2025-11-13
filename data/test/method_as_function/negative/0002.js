const axios = require('axios');
axios.get('https://petstore.swagger.io/v2/pets', {
    params: {
        foo: ['tag1', 'tag2'],
        bar: 10
    }
});