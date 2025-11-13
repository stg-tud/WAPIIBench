// This tests complex objects as array elements, only the first element is tested
const axios = require('axios');
axios.post('https://petstore.swagger.io/v2/customers', {
    name: 'name',
    addresses: [
        {
            city: "Darmstadt",
            house: {
                number: 1,
                street: "street"
            }
        }
    ]
});