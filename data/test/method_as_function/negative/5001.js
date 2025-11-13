// This tests objects that exceed the maximum recursion depth
const axios = require('axios');
axios.post('https://petstore.swagger.io/v2/customers', {
    name: 'name',
    addresses: [
        {
            city: "Darmstadt",
            house: {
                number: 1,
                street: "street",
                features: {
                    garages: 2,
                    pool: true,
                },
                badproperty: {}
            }
        }
    ]
});