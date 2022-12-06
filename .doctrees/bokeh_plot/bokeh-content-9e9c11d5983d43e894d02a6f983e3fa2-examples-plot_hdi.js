(function() {
  const fn = function() {
    (function(root) {
      function now() {
        return new Date();
      }
    
      const force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
    
    const element = document.getElementById("7741c9ab-8ed9-4cf2-8843-533a2477e246");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '7741c9ab-8ed9-4cf2-8843-533a2477e246' but no matching script tag was found.")
        }
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error(url) {
          console.error("failed to load " + url);
        }
    
        for (let i = 0; i < css_urls.length; i++) {
          const url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error.bind(null, url);
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        for (let i = 0; i < js_urls.length; i++) {
          const url = js_urls[i];
          const element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error.bind(null, url);
          element.async = false;
          element.src = url;
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      const js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-3.0.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-gl-3.0.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-3.0.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-3.0.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-mathjax-3.0.2.min.js"];
      const css_urls = [];
    
      const inline_js = [    function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        function(Bokeh) {
          (function() {
            const fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                  const docs_json = '{"c6fd93b0-57b3-4925-b955-4990730396cd":{"version":"3.0.2","title":"Bokeh Application","defs":[],"roots":[{"type":"object","name":"Figure","id":"p20858","attributes":{"width":500,"height":500,"x_range":{"type":"object","name":"DataRange1d","id":"p20859"},"y_range":{"type":"object","name":"DataRange1d","id":"p20860"},"x_scale":{"type":"object","name":"LinearScale","id":"p20871"},"y_scale":{"type":"object","name":"LinearScale","id":"p20873"},"title":{"type":"object","name":"Title","id":"p20862"},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p20916","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p20910","attributes":{"selected":{"type":"object","name":"Selection","id":"p20911","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p20912"},"data":{"type":"map","entries":[["x",{"type":"ndarray","array":{"type":"bytes","data":"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"},"shape":[400],"dtype":"float64","order":"little"}],["y",{"type":"ndarray","array":{"type":"bytes","data":"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"},"shape":[400],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p20917","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p20918"}}},"glyph":{"type":"object","name":"Patch","id":"p20913","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#ff0000","line_alpha":0,"fill_color":"#ff0000","fill_alpha":0.5}},"nonselection_glyph":{"type":"object","name":"Patch","id":"p20914","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#ff0000","line_alpha":0.1,"fill_color":"#ff0000","fill_alpha":0.1,"hatch_alpha":0.1}},"muted_glyph":{"type":"object","name":"Patch","id":"p20915","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#ff0000","line_alpha":0.2,"fill_color":"#ff0000","fill_alpha":0.2,"hatch_alpha":0.2}}}},{"type":"object","name":"GlyphRenderer","id":"p20925","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p20919","attributes":{"selected":{"type":"object","name":"Selection","id":"p20920","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p20921"},"data":{"type":"map","entries":[["x",{"type":"ndarray","array":{"type":"bytes","data":"pMMiPXhbBsBHss/mRogDwI6wkdYHxgLALrAY+t+dAsB9Pt9GtmQBwD9bAGy5zgDAZDtl3+tW/r88m44Nl8v7v4Uy550Xovu/1L7R8jaX97+7zbYjYUv3v8MljFWCiva/W3uWWPJY9r8taGUo7370v36cmLYtXPK/ZshgZUUL8b8Fp88BOvDwv0+Td2nV2vC/27z9kTmq8L+s9lNLEJHuvz+pOcJI5Ou/OGzR0QiK679ehYlib5rov+JHP17SLui/KylaY8sH6L832s62dW3gvyonNeyrrN+/u8IAe1ww3b/wL+27qFjavwjwJK0GnNm/MnwepAbl17+muCWKysTWvyeKCWE2wta/aXgg6lT41b928r7qLqPVv/TasHhmVdG/Y2Kh5rcq0b+MGlSvzSrNv0dcPDxFLcy/IJPKWJflyr/3tWeyq53IvxtQ8Vh/LsW/xmTtj4cqwr84hNh1CIa5vwrfnNF7c6W/4ssjF0fUbL8BfQFhGyunP3kYYFENjLE/QjgJXwPftD8hytp6Tfm4P9lTHvmkC8M/j/k1ekIWxD/XAtKYVJrHP5inazVGr8k/bQtF2+DSyT9a7SUglKLNPxdvhLM3pdA/eZxiR9kc0T8wHekty9TSP9VvnhQh7tI/8SmnDtiV1D+E8qIWjrrXP9MRyv4bwdc/XgVI7H1z2D+lxv1Vt93bPylxMYj98ds/Ar4O5x4Z3D8mjhHfKwTdP5jeA3k1Z94/kJ3wvmKx3z/6bcFHPQXgP2IhgbsoIOA/mGccJDTQ4D+j3fyUkOTiP4rTMBPhCOM/FkZWiLfn4z8MF6fuU/DkP+y+KNku0eU/vyxLBia75j8WgrdszHHnP/5hkJk6bOg/LLBbm09r6T8/4FrIT5fpP+rhPmBhmOo/VYnc2N+66j/OHIBekTnrPzrEkSSzwu0/4AkRCiOo8D8HNF1BI/byP0hcH+DtI/M/48/dxMYW9D9CtWms8031P5dWmWIVafk/kPAk1yaS+T+JSb94IeD6P5BM+B0cKgFAsz9kxYieAUC+rBX19NYGQF91QTY66gdAKhTFPFn7C0A="},"shape":[100],"dtype":"float64","order":"little"}],["y",{"type":"ndarray","array":{"type":"bytes","data":"uHi6hQ9J4z9ym2Aycu/oP+Se3FLwc+o/pJ/OC0DE6j8Gg0FykzbtP4JJ/yeNYu4/TmJNEIrU8D9isjh5NBryP75mDDH0LvI/liCXhmQ09D8imSRuT1r0Px7tOdW+uvQ/UsK004bT9D/qS81riMD1P8GxsyTp0fY/zZtPTV169z9+LBj/4of3P1g2REuVkvc/kiEBN+Oq9z9VAivtu1v4P7CVcc/tBvk/8qSLy30d+T+onl0nZNn5PwgucGhL9Pk/tXUpJw3++T9ySUySouT7PxtbeYJqCvw/qeefcPRZ/D8CWoLo6rT8P/9hWyp/zPw/ejB8K18D/T/rSLuuZif9P7vO3jO5J/0/8/C7YvVA/T+xIagimkv9P6Lk6TBT1f0/tNMrA6na/T9XvgolUy3+Pzw6PKwrPf4/zlZziqZR/j+hhNlEJXb+P/7qcAoYrf4/tCkBh1fd/j/eO1G8zzP/P4SMuRAyqv8/DTc67sr4/z/6AsI2Vi4AQGKARTUwRgBA4SR8DXxTAEApa+s15WMAQJ/yyCddmABAzK/RE7KgAEAXkMak0rwAQD1dqzF6zQBAWyjaBpfOAEBrLwGhFO0AQPFGOHtTCgFAyCl2lM0RAUDTkd6yTC0BQP3mSRHiLgFAn3LqgF1JAUAoL2rhqHsBQB2h7L8RfAFAVoDE3jeHAUBq3F91270BQBMXg9gfvwFA4Otw7pHBAUDiGPG9QtABQOo9kFdz5gFA2QnvKxb7AUC/LfiopwACQCwkcBcFBAJA84yDhAYaAkC0m58SklwCQHEaZiIcYQJAw8gK8fZ8AkDi4tR9Cp4CQN4XJdslugJAmGXJwGTXAkBD8JaNOe4CQEAMMlOHDQNABnZr82ktA0AIXAv56TIDQD3cBywMUwNAK5Eb+1tXA0CaA9ArMmcDQIc4kmRWuANAeEKEwggqBEACTVfQiL0EQBLXB3j7yARA+XM3sbEFBUBQbRrrfFMFQKZVplhFWgZAJDzJtYlkBkBi0i9eCLgGQEgm/A4OlQhA2h+yYkTPCEBf1op6emsLQLC6IBsd9QtAFYpinqz9DUA="},"shape":[100],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p20926","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p20927"}}},"glyph":{"type":"object","name":"Line","id":"p20922","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_width":3}},"nonselection_glyph":{"type":"object","name":"Line","id":"p20923","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_alpha":0.1,"line_width":3}},"muted_glyph":{"type":"object","name":"Line","id":"p20924","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_alpha":0.2,"line_width":3}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p20864","attributes":{"tools":[{"type":"object","name":"ResetTool","id":"p20889"},{"type":"object","name":"PanTool","id":"p20890"},{"type":"object","name":"BoxZoomTool","id":"p20891","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p20892","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"WheelZoomTool","id":"p20893"},{"type":"object","name":"LassoSelectTool","id":"p20894","attributes":{"renderers":"auto","overlay":{"type":"object","name":"PolyAnnotation","id":"p20895","attributes":{"syncable":false,"level":"overlay","visible":false,"xs":[],"xs_units":"canvas","ys":[],"ys_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"UndoTool","id":"p20896"},{"type":"object","name":"SaveTool","id":"p20897"},{"type":"object","name":"HoverTool","id":"p20898","attributes":{"renderers":"auto"}}]}},"toolbar_location":"above","left":[{"type":"object","name":"LinearAxis","id":"p20882","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p20885","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p20884"},"major_label_policy":{"type":"object","name":"AllLabels","id":"p20883"}}}],"below":[{"type":"object","name":"LinearAxis","id":"p20875","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p20878","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p20877"},"major_label_policy":{"type":"object","name":"AllLabels","id":"p20876"}}}],"center":[{"type":"object","name":"Grid","id":"p20881","attributes":{"axis":{"id":"p20875"}}},{"type":"object","name":"Grid","id":"p20888","attributes":{"dimension":1,"axis":{"id":"p20882"}}}],"output_backend":"webgl"}}]}}';
                  const render_items = [{"docid":"c6fd93b0-57b3-4925-b955-4990730396cd","roots":{"p20858":"7741c9ab-8ed9-4cf2-8843-533a2477e246"},"root_ids":["p20858"]}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    let attempts = 0;
                    const timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
    function(Bokeh) {
        }
      ];
    
      function run_inline_js() {
        for (let i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();